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

package org.apache.mahout.cf.taste.sgd.gradient;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

import static org.easymock.EasyMock.*;

public class TestRegularizedOrdinalGradient extends TasteTestCase{
  private RegularizedOrdinalGradient gradient;
  private Hypothesis hypothesis = createMock(Hypothesis.class);

  @Before
  public void setup(){
    gradient = new RegularizedOrdinalGradient(hypothesis, 1);
  }

  @Test
  public void testCumulativeProbs(){
    double[] prediction = new double[]{0.1, 0.4, 0.2, 0.3};
    double[] expectedCumulative = new double[]{0.1, 0.5, 0.7, 1.0};
    assertArrayEquals(expectedCumulative, gradient.cumulativeProbs(prediction), 0);
  }

  @Test
  public void testCutsGradient(){
    double y = 2;
    double[] cuts = {0.1, 0.2, 0.1};
    double[] linearCombination = {2,3,4};

    expect(hypothesis.predict(linearCombination)).andReturn(new double[]{0.1, 0.2, 0.6, 0.1});
    expect(hypothesis.predict(linearCombination)).andReturn(new double[]{0.1, 0.2, 0.6, 0.1});

    double probc = 0.6;

    double[] expectedGradient = new double[3];

    expectedGradient[0] = (1/probc) * (0.9*(1-0.9)*(1) - 0.3*(1-0.3)*1) - 0.001*0.1;
    expectedGradient[1] = (1/probc) * (0.9*(1-0.9)*(1) - 0.3*(1-0.3)*1) - 0.001*0.2;
    expectedGradient[2] = (1/probc) * (0.9*(1-0.9)*(1) - 0) - 0.001*0.1;
    replay(hypothesis);

    assertArrayEquals(expectedGradient, gradient.cutsGradient(y, cuts, linearCombination, 0.001), 0.0000001);
  }

  @Test
  public void testGradient(){
    double[] featureVals = {2,3};
    Vector feature = new DenseVector(featureVals);
    Vector[] features = new Vector[]{feature};

    double[] gradientWithLengthOneParam = gradient.gradient(0, 2, features, feature, new double[]{0.3, 0.6, 0.1}, 0.001);
    double[] gradientWithParamArray = gradient.gradient(0, 2, features, features, new double[]{0.3, 0.6, 0.1}, 0.001);
    assertArrayEquals(gradientWithLengthOneParam, gradientWithParamArray, 0);

    double[] expectedGradient = new double[1];
    expectedGradient[0] = 10 * ((1 * 0 * (-2)) - (0.9 * (1 - 0.9) * (-2)));
    assertArrayEquals(expectedGradient, gradientWithLengthOneParam, 0.000001);
    assertArrayEquals(expectedGradient, gradientWithParamArray, 0.000001);


    gradientWithLengthOneParam = gradient.gradient(1, 2, features, feature, new double[]{0.3, 0.6, 0.1}, 0.001);
    gradientWithParamArray = gradient.gradient(1, 2, features, features, new double[]{0.3, 0.6, 0.1}, 0.001);

    expectedGradient[0] = 10 * ((1 * 0 * (-3)) - (0.9 * (1 - 0.9) * (-3))) - 0.001 * 3;
    assertArrayEquals(expectedGradient, gradientWithLengthOneParam, 0.000001);
    assertArrayEquals(expectedGradient, gradientWithParamArray, 0.000001);

  }

}
