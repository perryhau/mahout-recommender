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

import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * Hypothesis for Ordinal Regression (Proportional Odds Model) based Recommender Systems
 */
public class OrdinalHypothesis extends Hypothesis {

  @Override
  public double[] predict(Vector[] x, Vector[] theta) {
    throw new UnsupportedOperationException();
  }

  @Override
  public double[] predict(double[] linearCombination) {
    double[] prediction = new double[linearCombination.length+1];
    double[] linearCombinationLogistic = new double[linearCombination.length];
    for(int i = 0; i<linearCombinationLogistic.length; i++){
      linearCombinationLogistic[i] = 1 / (1 + Math.exp(-linearCombination[i]));
    }

    prediction[0] = linearCombinationLogistic[0];
    for(int i = 1; i<prediction.length-1; i++){
      prediction[i]=linearCombinationLogistic[i]-linearCombinationLogistic[i-1];
    }
    prediction[prediction.length-1] = 1-linearCombinationLogistic[prediction.length-2];

    return prediction;
  }

  @Override
  public double[] predictFull(double[] linearCombination) {
    return predict(linearCombination);
  }

  /**
   * given ordinal cuts, user and item factors, and side info params and features, returns the linearCombination
   * @param cuts ordinal cuts
   * @param factors user and item factors
   * @param others side info vectors and corresponding params
   * @return threshold[i] - sum of dot products, where i is from 0 to numberOfCategories-2
   * Note that threshold[i] = threshold[i-1]+cuts[i]
   */
  @Override
  public double[] linearCombination(double[] cuts, Pair<Vector[], Vector[]> factors, List<Pair<Vector[], Vector>> others) {
    double[] linearCombination = new double[cuts.length];
    double dot = factors.getFirst()[0].dot(factors.getSecond()[0]);
    for(Pair<Vector[], Vector> other:others){
      dot += other.getFirst()[0].dot(other.getSecond());
    }
    double[] thresholds = new double[cuts.length];
    thresholds[0] = cuts[0];
    for(int i = 1; i<thresholds.length; i++){
      thresholds[i]=thresholds[i-1]+cuts[i];
    }
    for(int i = 0; i<linearCombination.length; i++){
      linearCombination[i] = thresholds[i]-dot;
    }
    return linearCombination;
  }

  /**
   * given ordinal cuts and user and item factors, returns the linearCombination
   * @param cuts ordinal cuts
   * @param alphas user factors
   * @param betas item factors
   * @return threshold[i] - sum of dot products, where i is from 0 to numberOfCategories-2
   * Note that threshold[i] = threshold[i-1]+cuts[i]
   */
  @Override
  public double[] linearCombination(double[] cuts, Vector[] alphas, Vector[] betas) {
    double[] linearCombination = new double[cuts.length];
    double dot = alphas[0].dot(betas[0]);
    double[] thresholds = new double[cuts.length];
    thresholds[0] = cuts[0];
    for(int i = 1; i<thresholds.length; i++){
      thresholds[i]=thresholds[i-1]+cuts[i];
    }
    for(int i = 0; i<linearCombination.length; i++){
      linearCombination[i] = thresholds[i]-dot;
    }
    return linearCombination;
  }
}
