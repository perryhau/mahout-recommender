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

package org.apache.mahout.cf.taste.sgd.hypothesis;

/**
 * Hypothesis for Logistic Regression based Recommender Systems
 */
public class LogisticHypothesis extends Hypothesis {

  @Override
  /**
   * {@inheritDoc}
   * computes the probability that the instance belongs to class 1.
   */
  public double[] predict(double[] linearCombination){
    return new double[]{1.0 / (1.0 + Math.exp(-1 * linearCombination[0]))};
  }

  @Override
  public double[] predictFull(double[] linearCombination) {
    double[] distr = new double[2];
    distr[1] = predict(linearCombination)[0];
    distr[0] = 1-distr[1];
    return distr;
  }
}
