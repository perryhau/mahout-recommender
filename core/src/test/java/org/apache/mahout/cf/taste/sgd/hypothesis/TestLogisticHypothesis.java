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

package org.apache.mahout.cf.taste.sgd.hypothesis;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Before;
import org.junit.Test;


public class TestLogisticHypothesis extends TasteTestCase {
  Hypothesis hypothesis;
  Vector[] x, theta;
  @Before
  public void setUp(){
    hypothesis = new LogisticHypothesis();
  }
  @Test
  public void predict(){
    Vector x1 = new DenseVector(new double[]{1,2,3});
    Vector x2 = new DenseVector(new double[]{1,4,3});
    Vector theta1 = new DenseVector(new double[]{3,2,1});
    Vector theta2 = new DenseVector(new double[]{3,4,1});
    x = new Vector[]{x1};

    theta = new Vector[] {theta1};
    assertArrayEquals(new double[]{1}, hypothesis.predict(x, theta),0.02);
    assertArrayEquals(hypothesis.predict(x, theta), hypothesis.predict(new double[]{10}), 0.001);
    x = new Vector[]{x1, x2};
    try{
      hypothesis.predict(x, theta);
      assertTrue(false);
    }
    catch(IllegalArgumentException e){
      assertTrue(true);
    }
    theta = new Vector[] {theta1, theta2};
    assertArrayEquals(new double[]{1}, hypothesis.predict(x, theta), 0.02);
    assertArrayEquals(hypothesis.predict(x, theta), hypothesis.predict(new double[] {10}), 0.001);


    Vector x3 = new RandomAccessSparseVector(3);
    x3.setQuick(0,1);
    x3.setQuick(3,2);
    x3.setQuick(4,3);

    Vector t3 = new RandomAccessSparseVector(3);
    t3.setQuick(0,3);
    t3.setQuick(3,2);
    t3.setQuick(4,1);
    x = new Vector[]{x3};
    theta = new Vector[]{t3};

    assertArrayEquals(new double[]{1}, hypothesis.predict(x, theta), 0.02);
    assertArrayEquals(hypothesis.predict(x, theta), hypothesis.predict(new double[]{10}), 0.001);

    x = new Vector[]{x1, x3};
    theta = new Vector[]{theta1, t3};
    assertArrayEquals(new double[]{1}, hypothesis.predict(x, theta), 0.02);
    assertArrayEquals(hypothesis.predict(x, theta), hypothesis.predict(new double[]{10, 10}), 0.001);

    x3.setQuick(8, 2);
    x = new Vector[]{x3};
    theta = new Vector[]{t3};
    assertArrayEquals(new double[]{1}, hypothesis.predict(x, theta), 0.02);
    assertArrayEquals(hypothesis.predict(x, theta), hypothesis.predict(new double[]{10, 10}), 0.001);

    Vector x4 = new RandomAccessSparseVector(0);
    x4.setQuick(3, 2);
    x4.setQuick(4, 1);
    x4.setQuick(12, 3);

    Vector t4 = new RandomAccessSparseVector(0);
    t4.setQuick(2, 2);
    t4.setQuick(4, 4);
    t4.setQuick(22, 1);
    t4.setQuick(10, 2);

    x = new Vector[]{x4};
    theta = new Vector[]{t4};
    assertArrayEquals(new double[]{1}, hypothesis.predict(x, theta), 0.02);
    assertArrayEquals(hypothesis.predict(x, theta), hypothesis.predict(new double[]{4}), 0.001);

    x4.assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return arg1*-1;
      }
    });

    assertArrayEquals(new double[]{0}, hypothesis.predict(x, theta), 0.02);
    assertArrayEquals(hypothesis.predict(x, theta), hypothesis.predict(new double[]{-4}), 0.001);



  }


}
