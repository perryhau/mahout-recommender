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

package org.apache.mahout.cf.taste.sgd.gradient;

import org.apache.mahout.math.Vector;

/**
 * The interface for classes to implement derivatives of the likelihood (or negative cost) function of one training instance with respect to specified index.
 */
public abstract class StochasticGradient {

  /**
   * Computes the derivative of the likelihood (or  negative cost) function with respect to <b>index</b>th element of a factor vector
   * The reason that the input factor vectors are arrays is to achieve robustness in case this is the gradient of a Softmax Regression.
   * @param index to compute derivative with respect to <b>index</b>th element of alphaI
   * @param y Response variable of the training instance
   * @param alphaI Array of factor vectors; one element for each class if this is a {@link RegularizedSoftmaxGradient}, otherwise alphaI[0] is the only element to consider
   * @param betaJ Array of factor vectors; one element for each class if this is a {@link RegularizedSoftmaxGradient}, betaJ[0] is the only element to consider otherwise
   * @param lambda Regularization rate
   * @return array of derivatives for each class
   */
  public abstract double[] gradient(int index,  double y, Vector[] alphaI,  Vector[] betaJ, double lambda);

  /**
   * Computes the derivative of the likelihood (or negative cost) function with respect to <b>index</b>th element of a parameter vector
   * The reason that the parameter vectors are arrays is to achieve robustness in case this is the gradient of a Softmax Regression.
   * @param index to compute derivative with respect to <b>index</b>th element of the parameter vector
   * @param y Response variable of the training instance
   * @param parameter Parameter vector to be updated, one vector for each class
   * @param feature Feature vector
   * @param prediction Prediction of {@link org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis} for that particular data instance
   * @param lambda Regularization rate, lambda is not used if index is 0 (intercept)
   * @return array of derivatives for each class
   */
  public abstract double[] gradient(int index, double y, Vector[] parameter, Vector feature, double[] prediction, double lambda);

  /**
   * Computes the derivative of the likelihood (or negative cost) function with respect to <b>index</b>th element of a parameter vector
   * The reason that the parameter vectors are arrays is to achieve robustness in case this is the gradient of a Softmax Regression.
   * @param index to compute derivative with respect to <b>index</b>th element of the parameter vector
   * @param y Response variable of the training instance
   * @param parameter Parameter vector to be updated, one vector for each class
   * @param feature Feature vector, one vector for each class
   * @param prediction Prediction of {@link org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis} for that particular data instance
   * @param lambda Regularization rate, lambda is not used if index is 0 (intercept)
   * @return array of derivatives for each class
   */
  public abstract double[] gradient(int index, double y, Vector[] parameter, Vector[] feature, double[] prediction, double lambda);
  public abstract void setLambda(double lambda);

  /**
   * Computes the derivatives to update the cuts in ordinal regression based recommender. The default implementation returns 0, as it is meaningless for recommenders
   * other than ordinal
   * @param y Response variable of the training instance
   * @param cuts Original cut values
   * @param linearCombination The linear combination without applying the ordinal link (see {@link org.apache.mahout.cf.taste.sgd.hypothesis.OrdinalHypothesis#linearCombination(double[], org.apache.mahout.common.Pair, java.util.List)})
   * @param lambda Regularization rate
   * @return gradient for each cut.
   */
  public double[] cutsGradient(double y, double[] cuts, double[] linearCombination, double lambda){
    double[] gradient = new double[cuts.length];
    for(int i = 0; i<gradient.length; i++){
      gradient[i] = 0;
    }
    return gradient;
  }

}
