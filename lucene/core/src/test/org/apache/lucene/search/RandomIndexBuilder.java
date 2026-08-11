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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.FieldSpec.FieldKind;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BytesRef;

/**
 * Builds a random Lucene index whose layout exercises the bulk-scorer specializations that are
 * gated by thresholds: the 2048-doc DV block boundary, {@code DenseConjunctionBulkScorer}'s
 * 1/32-density window, and {@code BooleanScorer}'s 2048-doc bucket table.
 */
class RandomIndexBuilder {

  static final int[] INTERESTING_DOC_COUNTS = {1, 512, 1024, 2047, 2048, 2049, 4095, 4096, 4097, 8192};

  private final Random random;

  RandomIndexBuilder(Random random) {
    this.random = random;
  }

  List<FieldSpec> build(Directory dir) throws IOException {
    int numDocs = chooseDocCount();
    int lowCard = 2 + random.nextInt(6);
    int highCard = Math.max(lowCard + 1, numDocs / 4);
    List<FieldSpec> specs = buildSpecs(lowCard, highCard);

    try (IndexWriter w = new IndexWriter(dir, new IndexWriterConfig())) {
      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        for (FieldSpec spec : specs) {
          addFieldValue(doc, spec, i);
        }
        w.addDocument(doc);
      }

      if (random.nextBoolean() && numDocs > 1) {
        int toDelete = 1 + random.nextInt(Math.min(numDocs / 4, 10));
        for (int i = 0; i < toDelete; i++) {
          w.deleteDocuments(
              new Term(
                  specs.get(0).name(),
                  specs.get(0).terms().get(random.nextInt(specs.get(0).terms().size()))));
        }
      }

      if (random.nextBoolean()) {
        w.forceMerge(1);
      }
    }

    return specs;
  }

  private int chooseDocCount() {
    if (random.nextInt(3) == 0) {
      return INTERESTING_DOC_COUNTS[random.nextInt(INTERESTING_DOC_COUNTS.length)];
    }
    return 1 + random.nextInt(8192);
  }

  private List<FieldSpec> buildSpecs(int lowCard, int highCard) {
    List<FieldSpec> specs = new ArrayList<>();
    specs.add(FieldSpec.keyword("f_kw", FieldKind.INDEXED_KEYWORD, makeTerms("kw", lowCard)));
    specs.add(FieldSpec.keyword("f_dv_kw", FieldKind.DV_KEYWORD, makeTerms("dv", lowCard)));
    specs.add(FieldSpec.numeric("f_dv_num", FieldKind.DV_NUMERIC, 0L, highCard - 1L));
    specs.add(FieldSpec.numeric("f_pt", FieldKind.POINT_LONG, 0L, highCard - 1L));
    return specs;
  }

  private List<BytesRef> makeTerms(String prefix, int n) {
    List<BytesRef> terms = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      terms.add(new BytesRef(prefix + "_" + i));
    }
    return terms;
  }

  private void addFieldValue(Document doc, FieldSpec spec, int docId) {
    switch (spec.kind()) {
      case INDEXED_KEYWORD -> {
        BytesRef term = spec.terms().get(docId % spec.terms().size());
        doc.add(new StringField(spec.name(), term.utf8ToString(), Field.Store.NO));
      }
      case DV_KEYWORD -> {
        BytesRef term = spec.terms().get(docId % spec.terms().size());
        if (random.nextInt(5) != 0) {
          // indexedField enables the DocValues skip index (RANGE type), which is required for
          // DocValuesBlockRangeIterator to be constructed and for its docIDRunEnd() to be called.
          doc.add(SortedSetDocValuesField.indexedField(spec.name(), term));
        }
      }
      case DV_NUMERIC -> {
        long val = spec.minValue() + (docId % (spec.maxValue() - spec.minValue() + 1));
        if (random.nextInt(5) != 0) {
          doc.add(new SortedNumericDocValuesField(spec.name(), val));
        }
      }
      case POINT_LONG -> {
        long val = spec.minValue() + (docId % (spec.maxValue() - spec.minValue() + 1));
        doc.add(new LongPoint(spec.name() + "_point", val));
        doc.add(new SortedNumericDocValuesField(spec.name() + "_dv", val));
      }
    }
  }
}
