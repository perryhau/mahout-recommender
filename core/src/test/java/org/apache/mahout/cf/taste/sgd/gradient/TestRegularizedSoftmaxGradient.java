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
import org.apache.mahout.cf.taste.sgd.hypothesis.SoftmaxHypothesis;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

import static org.easymock.EasyMock.*;

public class TestRegularizedSoftmaxGradient extends TasteTestCase {
  private StochasticGradient gradientSoftmax;

  @Before
  public void setUp(){
    gradientSoftmax = new RegularizedSoftmaxGradient(new SoftmaxHypothesis(), 0.1);
  }

  @Test
  public void gradient(){
    Vector[] alphas = new Vector[3];
    alphas[0] = new DenseVector(new double[] {1,2});
    alphas[1] = new DenseVector(new double[] {3,4});
    alphas[2] = new DenseVector(new double[] {5,6});

    Vector[] betas = new Vector[3];
    betas[0] = new DenseVector(new double[] {2,1});
    betas[1] = new DenseVector(new double[] {4,3});
    betas[2] = new DenseVector(new double[] {6,5});

    Vector[] thetas_i = new Vector[3];
    thetas_i[0] = new RandomAccessSparseVector(2);
    thetas_i[0].setQuick(3, 2);
    thetas_i[0].setQuick(7, 1);

    thetas_i[1] = new RandomAccessSparseVector(2);
    thetas_i[1].setQuick(3, 1);
    thetas_i[1].setQuick(7, 4);

    thetas_i[2] = new RandomAccessSparseVector(2);
    thetas_i[2].setQuick(3, -1);
    thetas_i[2].setQuick(7, 1);


    Vector t_j = new RandomAccessSparseVector(2);
    t_j.setQuick(3, 2);
    t_j.setQuick(7, -2);

    double[] linearCombination = new double[3];
    linearCombination[0] = alphas[0].dot(betas[0])+thetas_i[0].dot(t_j);
    linearCombination[1] = alphas[1].dot(betas[1])+thetas_i[1].dot(t_j);
    linearCombination[2] = alphas[2].dot(betas[2])+thetas_i[2].dot(t_j);


    Hypothesis hypothesis = createMock(Hypothesis.class);
    expect(hypothesis.predict(linearCombination)).andReturn(new double[]{0.001,0.001,0.999}).times(3);
    replay(hypothesis);

    //actual class 0, params are alphas, features are betas, index is 1
    assertArrayEquals(new double[]{0.0979, -0.0043, -0.5055}, gradientSoftmax.gradient(1, 0, alphas, betas, hypothesis.predict(linearCombination), 0.01), 0.001);
    //actual class 1, params are thetas, features are thetas_i, index is 3
    assertArrayEquals(new double[]{-0.0022,0.1988,-0.1988}, gradientSoftmax.gradient(3, 1, thetas_i, t_j, hypothesis.predict(linearCombination), 0.01), 0.001);
    //actual class 2, params are betas, features are alphas, index is 0
    assertArrayEquals(new double[]{0,0,0.4001}, gradientSoftmax.gradient(0, 2, betas, alphas, hypothesis.predict(linearCombination), 0.01), 0.001);

    verify(hypothesis);
  }
}
