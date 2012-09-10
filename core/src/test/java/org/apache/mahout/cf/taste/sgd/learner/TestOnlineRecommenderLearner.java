/**
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

package org.apache.mahout.cf.taste.sgd.learner;

import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class TestOnlineRecommenderLearner {
  @Test
  public void getNonZeroIndices(){
    Vector vector = new RandomAccessSparseVector(2);
    vector.setQuick(3,3.4);
    vector.setQuick(5,2.2);
    vector.setQuick(1,3);
    vector.setQuick(100,1);
    vector.setQuick(10,2);

    ArrayList<Integer> list = OnlineRecommenderLearner.getNonZeroIndices(vector);
    assertTrue(5 == list.size());

    assertTrue(list.contains(1));
    assertTrue(list.contains(3));
    assertTrue(list.contains(5));
    assertTrue(list.contains(10));
    assertTrue(list.contains(100));
    assertFalse(list.contains(0));
    assertFalse(list.contains(2));
    assertFalse(list.contains(4));
    assertFalse(list.contains(101));

    vector.setQuick(10, 0);
    ArrayList<Integer> list1 = OnlineRecommenderLearner.getNonZeroIndices(vector);
    assertFalse(list1.contains(10));

  }
}
