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
import org.apache.mahout.cf.taste.sgd.hypothesis.LogisticHypothesis;
import org.apache.mahout.cf.taste.sgd.hypothesis.OLSHypothesis;
import org.apache.mahout.cf.taste.sgd.hypothesis.PoissonHypothesis;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.junit.Before;
import org.junit.Test;

import static org.easymock.EasyMock.*;

public class TestRegularizedDefaultGradient extends TasteTestCase{
  private StochasticGradient gradientOLS, gradientLogistic, gradientPoisson;

  @Before
  public void setUp(){

  }

  @Test
  public void gradient(){

    double olsActual = 20;
    double logisticActual = 0;
    double poissonActual = 10;

    Vector alpha_i = new DenseVector(new double[] {1,2,3});
    Vector beta_j = new DenseVector(new double[] {3,2,1});

    Vector theta_i = new RandomAccessSparseVector(3);
    Vector t_j = new RandomAccessSparseVector(3);

    theta_i.setQuick(0, 3);
    theta_i.setQuick(1, 4);
    theta_i.setQuick(5, 5);
    theta_i.setQuick(7, 2);

    t_j.setQuick(0, 1);
    t_j.setQuick(1, 2);
    t_j.setQuick(5, 4);
    t_j.setQuick(7, 3);

    Vector[] alphas = new Vector[]{alpha_i};
    Vector[] betas = new Vector[]{beta_j};


    Hypothesis h1 = createMock(OLSHypothesis.class);
    Hypothesis h2 = createMock(LogisticHypothesis.class);
    Hypothesis h3 = createMock(PoissonHypothesis.class);

    gradientOLS = new RegularizedDefaultGradient(h1, 0.1);
    gradientLogistic = new RegularizedDefaultGradient(h2, 0.1);
    gradientPoisson = new RegularizedDefaultGradient(h3, 0.1);

    expect(h1.predict(alphas, betas)).andReturn(new double[]{10});
    expect(h1.predict(betas, alphas)).andReturn(new double[]{30});
    expect(h2.predict(alphas, betas)).andReturn(new double[]{0.98});
    expect(h3.predict(alphas, betas)).andReturn(new double[]{Math.exp(10)});
    replay(h1, h2, h3);

    assertArrayEquals(new double[]{1}, gradientOLS.gradient(0, olsActual, alphas, betas, 0.01), 0.01);
    assertArrayEquals(new double[]{-1}, gradientOLS.gradient(0, olsActual, betas, alphas, 0.01), 0.01);
    assertArrayEquals(new double[]{1}, gradientOLS.gradient(0, olsActual, alphas, betas, new double[]{10}, 0.01), 0.01);
    assertArrayEquals(new double[]{-1.4}, gradientOLS.gradient(0, olsActual, new Vector[]{theta_i}, new Vector[]{t_j}, new double[]{34}, 0.01), 0.01);
    assertArrayEquals(new double[]{-5.605}, gradientOLS.gradient(5, olsActual, new Vector[]{theta_i}, new Vector[]{t_j}, new double[]{34}, 0.01), 0.01);
    assertArrayEquals(new double[]{0}, gradientOLS.gradient(5, olsActual, new Vector[]{theta_i}, new Vector[]{t_j}, new double[]{20},0.01), 0.01);

    assertArrayEquals(new double[]{-0.194}, gradientLogistic.gradient(1, logisticActual, alphas, betas, new double[]{0.98}, 0.01),  0.01);
    assertArrayEquals(new double[]{-0.194}, gradientLogistic.gradient(1, logisticActual, alphas, betas, 0.01), 0.01);
    assertArrayEquals(new double[]{-0.292}, gradientLogistic.gradient(7, logisticActual, new Vector[]{theta_i}, new Vector[]{t_j}, new double[]{0.98}, 0.01), 0.01);
    assertArrayEquals(new double[]{0}, gradientLogistic.gradient(7, logisticActual, new Vector[]{theta_i}, new Vector[]{t_j}, new double[]{0}, 0.01), 0.01);

    assertArrayEquals(new double[]{-2201.5}, gradientPoisson.gradient(2, poissonActual, alphas, betas, 0.01), 5);
    assertArrayEquals(new double[]{-2201.5}, gradientPoisson.gradient(2, poissonActual, alphas, betas, new double[]{Math.exp(10)}, 0.01), 5);
    assertArrayEquals(new double[]{-2.012}, gradientPoisson.gradient(1, poissonActual, new Vector[]{theta_i}, new Vector[]{t_j}, new double[]{Math.exp(3)}, 0.01), 0.01);
    assertArrayEquals(new double[]{0}, gradientPoisson.gradient(1, poissonActual, new Vector[]{theta_i}, new Vector[]{t_j}, new double[]{10}, 0.01), 0.01);

    verify(h1, h2, h3);

  }
}

