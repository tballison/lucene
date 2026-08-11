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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.util.BytesRef;

/**
 * Generates random Lucene {@link Query} trees for differential oracle testing.
 *
 * <p>Uses only the Lucene public API. Driven by a caller-supplied {@link Random}; a failing seed
 * replays exactly. Generates nested {@link BooleanQuery} nodes mixing all four {@link Occur}
 * values, doc-values-only leaf queries (which produce {@link TwoPhaseIterator}s), indexed queries,
 * {@link IndexOrDocValuesQuery}, and trivial sentinels.
 */
// newSlowExactQuery / newSlowRangeQuery are Lucene test-only APIs marked deprecated.
@SuppressWarnings("deprecation")
class RandomLuceneQueryGenerator {

  static final int MAX_DEPTH = 3;
  static final int MAX_CLAUSES = 5;

  private final Random random;
  private final List<FieldSpec> fields;

  RandomLuceneQueryGenerator(Random random, List<FieldSpec> fields) {
    this.random = random;
    this.fields = fields;
  }

  Query next() {
    // ~1/3 of top-level queries are pure FILTER conjunctions. This biases coverage toward
    // DenseConjunctionBulkScorer, which only fires for all-FILTER queries on large dense indices.
    if (random.nextInt(3) == 0) {
      return randomFilterConjunction();
    }
    return generate(MAX_DEPTH);
  }

  /** Generates a BooleanQuery where every clause uses {@link Occur#FILTER}. */
  private Query randomFilterConjunction() {
    int numClauses = 2 + random.nextInt(MAX_CLAUSES - 1);
    BooleanQuery.Builder b = new BooleanQuery.Builder();
    for (int i = 0; i < numClauses; i++) {
      b.add(randomLeaf(), Occur.FILTER);
    }
    return b.build();
  }

  private Query generate(int depth) {
    if (depth == 0 || (depth < MAX_DEPTH && random.nextInt(10) < 3)) {
      return randomLeaf();
    }
    return randomBool(depth);
  }

  private Query randomBool(int depth) {
    int numClauses = 1 + random.nextInt(MAX_CLAUSES);
    BooleanQuery.Builder b = new BooleanQuery.Builder();

    int mustNotCount = 0;
    List<Query> clauses = new ArrayList<>(numClauses);
    List<Occur> occurs = new ArrayList<>(numClauses);

    for (int i = 0; i < numClauses; i++) {
      clauses.add(generate(depth - 1));
      occurs.add(randomOccur());
      if (occurs.get(i) == Occur.MUST_NOT) {
        mustNotCount++;
      }
    }

    if (mustNotCount == numClauses) {
      occurs.set(random.nextInt(numClauses), Occur.FILTER);
    }

    int numShould = 0;
    for (int i = 0; i < numClauses; i++) {
      b.add(clauses.get(i), occurs.get(i));
      if (occurs.get(i) == Occur.SHOULD) {
        numShould++;
      }
    }

    if (numShould > 0) {
      b.setMinimumNumberShouldMatch(random.nextInt(numShould + 1));
    }

    return b.build();
  }

  private Occur randomOccur() {
    return switch (random.nextInt(4)) {
      case 0 -> Occur.MUST;
      case 1 -> Occur.FILTER;
      case 2 -> Occur.SHOULD;
      default -> Occur.MUST_NOT;
    };
  }

  private Query randomLeaf() {
    int roll = random.nextInt(20);
    if (roll == 0) return new MatchAllDocsQuery();
    if (roll == 1) return new MatchNoDocsQuery();
    if (roll == 2) return new ConstantScoreQuery(randomLeafFromField());
    return randomLeafFromField();
  }

  private Query randomLeafFromField() {
    FieldSpec f = fields.get(random.nextInt(fields.size()));
    return switch (f.kind()) {
      case INDEXED_KEYWORD -> randomIndexedKeywordLeaf(f);
      case DV_KEYWORD -> randomDvKeywordLeaf(f);
      case DV_NUMERIC -> randomDvNumericLeaf(f);
      case POINT_LONG -> randomPointLongLeaf(f);
    };
  }

  private Query randomIndexedKeywordLeaf(FieldSpec f) {
    return switch (random.nextInt(3)) {
      case 0 -> new TermQuery(new Term(f.name(), randomTerm(f)));
      case 1 -> {
        int n = 1 + random.nextInt(Math.min(3, f.terms().size()));
        List<BytesRef> set = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
          set.add(randomTerm(f));
        }
        yield new TermInSetQuery(f.name(), set);
      }
      default -> {
        BytesRef lo = randomTermOrNull(f);
        BytesRef hi = randomTermOrNull(f);
        if (lo != null && hi != null && lo.compareTo(hi) > 0) {
          BytesRef tmp = lo;
          lo = hi;
          hi = tmp;
        }
        yield new TermRangeQuery(f.name(), lo, hi, random.nextBoolean(), random.nextBoolean());
      }
    };
  }

  private Query randomDvKeywordLeaf(FieldSpec f) {
    return switch (random.nextInt(3)) {
      case 0 -> SortedSetDocValuesField.newSlowExactQuery(f.name(), randomTerm(f));
      case 1 -> {
        BytesRef lo = randomTerm(f);
        BytesRef hi = randomTerm(f);
        if (lo.compareTo(hi) > 0) {
          BytesRef tmp = lo;
          lo = hi;
          hi = tmp;
        }
        yield SortedSetDocValuesField.newSlowRangeQuery(f.name(), lo, hi, true, true);
      }
      default -> randomDvKeywordSetLeaf(f);
    };
  }

  /**
   * Picks every other term from the sorted vocabulary so there is always at least one ordinal gap,
   * forcing {@code DocValuesBlockRangeIterator} (the non-contiguous variant) rather than the bulk
   * contiguous variant. This exercises the {@code docIDRunEnd()} code path that had the PR #16450
   * bug.
   */
  private Query randomDvKeywordSetLeaf(FieldSpec f) {
    List<BytesRef> all = f.terms();
    if (all.size() < 3) {
      // Can't form a non-contiguous set with fewer than 3 terms; fall back to exact.
      return SortedSetDocValuesField.newSlowExactQuery(f.name(), randomTerm(f));
    }
    int start = random.nextInt(2); // vary the skip pattern across seeds
    List<BytesRef> subset = new ArrayList<>();
    for (int i = start; i < all.size(); i += 2) {
      subset.add(all.get(i));
    }
    if (subset.size() < 2) {
      subset.add(all.get(all.size() - 1));
    }
    return SortedSetDocValuesField.newSlowSetQuery(f.name(), subset);
  }

  private Query randomDvNumericLeaf(FieldSpec f) {
    if (random.nextBoolean()) {
      return SortedNumericDocValuesField.newSlowExactQuery(f.name(), randomLong(f));
    }
    long lo = randomLong(f);
    long hi = randomLong(f);
    if (lo > hi) {
      long tmp = lo;
      lo = hi;
      hi = tmp;
    }
    return SortedNumericDocValuesField.newSlowRangeQuery(f.name(), lo, hi);
  }

  private Query randomPointLongLeaf(FieldSpec f) {
    long lo = randomLong(f);
    long hi = randomLong(f);
    if (lo > hi) {
      long tmp = lo;
      lo = hi;
      hi = tmp;
    }
    Query pointQuery = LongPoint.newRangeQuery(f.name() + "_point", lo, hi);
    Query dvQuery = SortedNumericDocValuesField.newSlowRangeQuery(f.name() + "_dv", lo, hi);
    if (random.nextBoolean()) {
      return new IndexOrDocValuesQuery(pointQuery, dvQuery);
    }
    return random.nextBoolean() ? pointQuery : dvQuery;
  }

  private BytesRef randomTerm(FieldSpec f) {
    if (random.nextInt(5) == 0 || f.terms().isEmpty()) {
      return new BytesRef("zzz_outside_" + random.nextInt(100));
    }
    return f.terms().get(random.nextInt(f.terms().size()));
  }

  private BytesRef randomTermOrNull(FieldSpec f) {
    return random.nextInt(5) == 0 ? null : randomTerm(f);
  }

  private long randomLong(FieldSpec f) {
    if (random.nextInt(5) == 0) {
      return f.maxValue() + 1 + random.nextInt(10);
    }
    long range = f.maxValue() - f.minValue() + 1;
    return f.minValue() + (range == 0 ? 0 : (random.nextLong() & Long.MAX_VALUE) % range);
  }
}
