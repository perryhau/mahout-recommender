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

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

public class TestOrdinalHypothesis extends TasteTestCase{
  private OrdinalHypothesis hypothesis;
  @Before
  public void setup(){
    hypothesis = new OrdinalHypothesis();
  }
  @Test
  public void testLinearCombination(){
    double[] cuts = new double[]{0.1,0.5,0.2};
    Vector alphas = new DenseVector(2);
    alphas.setQuick(0, 1);
    alphas.setQuick(1, 4);
    Vector betas = new DenseVector(2);
    betas.setQuick(0, 3);
    betas.setQuick(1, 2);

    Vector thetasOnT = new RandomAccessSparseVector(0);
    thetasOnT.setQuick(0,2);
    Vector t = new RandomAccessSparseVector(0);
    t.setQuick(0,3);
    t.setQuick(1,3);

    double[] sideInfoResults = new double[]{-16.9,-16.4,-16.2};
    double[] factorsResults = new double[]{-10.9,-10.4,-10.2};


    double[] sideInfoPrediction = hypothesis.linearCombination(cuts, Pair.of(new Vector[]{alphas}, new Vector[]{betas}), Lists.newArrayList(Pair.of(new Vector[]{thetasOnT}, t)));
    double[] factorsPrediction =  hypothesis.linearCombination(cuts, new Vector[]{alphas}, new Vector[]{betas});

    assertArrayEquals(sideInfoResults, sideInfoPrediction, 0);
    assertArrayEquals(factorsResults, factorsPrediction, 0);

  }

  @Test
  public void testClassDistribution(){
    double[] linearCombination = new double[]{2,3,4};
    double class1 = 1/(1+Math.exp(-2));
    double class2 = 1/(1+Math.exp(-3)) - class1;
    double class3 = 1/(1+Math.exp(-4)) - (class1+class2);
    double class4 = 1 - (class1+class2+class3);

    double[] expectedClassDistribution = new double[]{class1, class2, class3, class4};
    System.out.println(Arrays.toString(expectedClassDistribution));
    double[] actualClassDistribution = hypothesis.predict(linearCombination);
    assertArrayEquals(expectedClassDistribution, actualClassDistribution, 0.01);

  }

}
