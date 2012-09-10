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

import com.google.common.base.Preconditions;

/**
 * Hypothesis for Softmax Regression based Recommender Systems
 */
public class SoftmaxHypothesis extends Hypothesis {

  @Override
  public double[] predict(double[] linearCombination) {
    Preconditions.checkNotNull(linearCombination);
    Preconditions.checkArgument(linearCombination.length > 2);
    double[] probs = new double[linearCombination.length];
    double denom = 0;

    for (int c = 0; c<linearCombination.length; c++){
      probs[c] = Math.exp(linearCombination[c]);
      denom += probs[c];
    }
    for (int c = 0; c<linearCombination.length; c++){
      probs[c] /= denom;
    }
    return probs;
  }

  @Override
  public double[] predictFull(double[] linearCombination) {
    return predict(linearCombination);
  }
}
