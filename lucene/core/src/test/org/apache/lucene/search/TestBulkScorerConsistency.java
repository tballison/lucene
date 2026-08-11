/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.search;

import java.io.IOException;
import java.util.List;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.search.FixedBitSetCollector;
import org.apache.lucene.tests.search.ScorerIndexSearcher;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;

/**
 * Differential oracle: asserts that Lucene's optimised bulk-scoring path produces exactly the same
 * document set as the naive doc-by-doc {@link Scorer} path, for randomly generated queries over
 * randomly built indexes.
 *
 * <p>This test was written to verify that the harness catches the Lucene 9.5.0 bug (PR #16450) in
 * which {@code DocValuesRangeIterator.docIDRunEnd()} over-reported run boundaries for {@code MAYBE}
 * / {@code YES_IF_PRESENT} blocks, causing {@code ReqExclBulkScorer} to skip per-doc {@code
 * matches()} calls and return false positives for {@code must_not} queries over doc-values-only
 * fields. Run this test at the pre-fix commit to confirm the failure; run at or after the fix to
 * confirm it passes.
 */
public class TestBulkScorerConsistency extends LuceneTestCase {

  /**
   * Main fuzz test: random index, random queries, asserts fast == slow.
   *
   * <p>The fast searcher is a bare {@link IndexSearcher} (not {@code newSearcher()}) so that
   * {@link org.apache.lucene.tests.search.AssertingScorer} does not mask {@code docIDRunEnd()}
   * bugs by falling back to the conservative default half the time.
   */
  public void testBulkEqualsScorer() throws IOException {
    try (Directory dir = newDirectory()) {
      List<FieldSpec> fields = new RandomIndexBuilder(random()).build(dir);
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher fastSearcher = new IndexSearcher(reader);
        ScorerIndexSearcher slowSearcher = new ScorerIndexSearcher(reader);
        fastSearcher.setQueryCache(null);
        slowSearcher.setQueryCache(null);

        RandomLuceneQueryGenerator gen = new RandomLuceneQueryGenerator(random(), fields);
        int numQueries = 20 + random().nextInt(81);
        for (int q = 0; q < numQueries; q++) {
          assertBulkEqualsScorer(fastSearcher, slowSearcher, gen.next());
        }
      }
    }
  }

  /**
   * Self-test: proves the harness detects a broken bulk scorer. {@link LyingIndexSearcher}
   * collects every doc without checking matches — same class of mistake as the 9.5.0 bug.
   */
  public void testHarnessDetectsBrokenBulkScorer() throws IOException {
    try (Directory dir = newDirectory()) {
      try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
        for (int i = 0; i < 100; i++) {
          Document doc = new Document();
          doc.add(new StringField("f", i % 2 == 0 ? "even" : "odd", Field.Store.NO));
          w.addDocument(doc);
        }
      }

      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        ScorerIndexSearcher correct = new ScorerIndexSearcher(reader);
        LyingIndexSearcher lying = new LyingIndexSearcher(reader);
        correct.setQueryCache(null);
        lying.setQueryCache(null);

        int maxDoc = reader.maxDoc();
        Query query = new TermQuery(new Term("f", "even"));

        FixedBitSet correctResult = correct.search(query, FixedBitSetCollector.createManager(maxDoc));
        FixedBitSet lyingResult = lying.search(query, FixedBitSetCollector.createManager(maxDoc));

        assertEquals(50, correctResult.cardinality());
        assertFalse(lyingResult.equals(correctResult));
        assertEquals(maxDoc, lyingResult.cardinality());
      }
    }
  }

  /**
   * Targeted reproduction of apache/lucene#16450: {@code DocValuesBlockRangeIterator.docIDRunEnd()}
   * returned {@code doc+1} unconditionally. For a {@code WINDOW_SIZE+1}-doc index the trailing
   * single-doc window has {@code minRunEndThreshold = max = doc+1}, making the non-contiguous
   * ordinal-set clause a "non-window" clause. {@code DenseConjunctionBulkScorer} then fires
   * {@code collectRange} without calling {@code TwoPhaseIterator.matches()}, collecting a false
   * positive.
   *
   * <p>This test is expected to FAIL at the pre-fix commit ({@code c93628f669}) and PASS after the
   * fix ({@code aeb84c0346}).
   *
   * <p>Note: the fast searcher must be a bare {@link IndexSearcher}, NOT {@code newSearcher()}.
   * {@code newSearcher()} returns an {@link
   * org.apache.lucene.tests.search.AssertingIndexSearcher} whose {@code AssertingScorer} wrappers
   * override the TPI with an anonymous class that does NOT override {@code docIDRunEnd()}, so the
   * conservative default ({@code approximation().docID()}) is always returned, preventing the
   * {@code collectRange} short-circuit from firing even on the buggy code.
   */
  public void testCatchesDVOrdinalSetFalsePositive() throws IOException {
    int maxDoc = DenseConjunctionBulkScorer.WINDOW_SIZE + 1; // 4097
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new Lucene104Codec());
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < maxDoc; i++) {
          Document doc = new Document();
          // Most docs: "aaa" (ord 0). Every 100th: "ccc" (ord 2) to establish the 3-term vocab.
          // Last doc: "bbb" (ord 1) — inside bounding range [0,2] but NOT in set {0,2}.
          String val = (i == maxDoc - 1) ? "bbb" : (i % 100 == 0 ? "ccc" : "aaa");
          doc.add(SortedDocValuesField.indexedField("dv", new BytesRef(val)));
          w.addDocument(doc);
        }
        w.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher fast = new IndexSearcher(reader);
        ScorerIndexSearcher slow = new ScorerIndexSearcher(reader);
        fast.setQueryCache(null);
        slow.setQueryCache(null);

        // Non-contiguous ordinal set {aaa=0, ccc=2}: gap at bbb=1 forces
        // DocValuesBlockRangeIterator (not the Bulk variant), whose pre-fix docIDRunEnd()
        // returned doc+1 — triggering the false-positive path in DenseConjunctionBulkScorer.
        Query setQuery =
            SortedDocValuesField.newSlowSetQuery(
                "dv", List.of(new BytesRef("aaa"), new BytesRef("ccc")));
        Query rangeQuery =
            SortedDocValuesField.newSlowRangeQuery(
                "dv", new BytesRef("aaa"), new BytesRef("ccc"), true, true);
        Query q =
            new BooleanQuery.Builder()
                .add(setQuery, BooleanClause.Occur.FILTER)
                .add(rangeQuery, BooleanClause.Occur.FILTER)
                .build();

        FixedBitSet fastBits = fast.search(q, FixedBitSetCollector.createManager(maxDoc));
        FixedBitSet slowBits = slow.search(q, FixedBitSetCollector.createManager(maxDoc));
        // If the bug triggers: fast has maxDoc matches (false positive), slow has maxDoc-1.
        // Expect FAIL at pre-fix commit c93628f669; PASS at or after fix aeb84c0346.
        assertEquals(
            "fast path cardinality (bug → " + maxDoc + ", correct → " + (maxDoc - 1) + ")",
            maxDoc - 1,
            fastBits.cardinality());
        assertEquals(
            "slow path cardinality (bug → " + maxDoc + ", correct → " + (maxDoc - 1) + ")",
            maxDoc - 1,
            slowBits.cardinality());
      }
    }
  }

  static void assertBulkEqualsScorer(
      IndexSearcher fastSearcher, ScorerIndexSearcher slowSearcher, Query query)
      throws IOException {
    int maxDoc = fastSearcher.getIndexReader().maxDoc();
    FixedBitSet fast = fastSearcher.search(query, FixedBitSetCollector.createManager(maxDoc));
    FixedBitSet slow = slowSearcher.search(query, FixedBitSetCollector.createManager(maxDoc));
    assertEquals("bulk scorer and doc-by-doc scorer disagree on: " + query, slow, fast);
  }

  static class LyingIndexSearcher extends IndexSearcher {
    LyingIndexSearcher(org.apache.lucene.index.IndexReader r) {
      super(r);
    }

    @Override
    protected void searchLeaf(
        LeafReaderContext ctx, int minDocId, int maxDocId, Weight weight, Collector collector)
        throws IOException {
      final LeafCollector leafCollector;
      try {
        leafCollector = collector.getLeafCollector(ctx);
      } catch (CollectionTerminatedException e) {
        return;
      }
      ScorerSupplier ss = weight.scorerSupplier(ctx);
      if (ss == null) {
        leafCollector.finish();
        return;
      }
      BulkScorer real = ss.bulkScorer();
      if (real == null) {
        leafCollector.finish();
        return;
      }
      try {
        int maxDoc = Math.min(maxDocId, ctx.reader().maxDoc());
        new LyingBulkScorer(real).score(leafCollector, ctx.reader().getLiveDocs(), minDocId, maxDoc);
      } catch (CollectionTerminatedException e) {
        // normal
      }
      leafCollector.finish();
    }
  }

  static class LyingBulkScorer extends BulkScorer {
    private final BulkScorer in;

    LyingBulkScorer(BulkScorer in) {
      this.in = in;
    }

    @Override
    public int score(LeafCollector collector, Bits acceptDocs, int min, int max)
        throws IOException {
      collector.setScorer(
          new Scorable() {
            @Override
            public float score() {
              return 0f;
            }
          });
      for (int doc = min; doc < max; doc++) {
        if (acceptDocs == null || acceptDocs.get(doc)) {
          collector.collect(doc);
        }
      }
      return max;
    }

    @Override
    public long cost() {
      return in.cost();
    }
  }
}
