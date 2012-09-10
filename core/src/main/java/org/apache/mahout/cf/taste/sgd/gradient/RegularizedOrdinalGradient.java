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

import org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@inheritDoc}
 * Gradient implementation for Ordinal Recommendation
 */
public class RegularizedOrdinalGradient extends StochasticGradient{
  private Hypothesis hypothesis;
  private double learningRate;
  Logger logger = LoggerFactory.getLogger(RegularizedOrdinalGradient.class);

  public RegularizedOrdinalGradient(Hypothesis hypothesis, double learningRate) {
    this.hypothesis = hypothesis;
    this.learningRate = learningRate;
    logger.info("Using ordinal gradient with learning rate: "+learningRate);
  }

  @Override
  public double[] gradient(int index, double y, Vector[] alphaI, Vector[] betaJ, double lambda) {
    throw new UnsupportedOperationException();
  }

  @Override
  public double[] gradient(int index, double y, Vector[] parameter, Vector feature, double[] prediction, double lambda) {
    Vector[] features = new Vector[parameter.length];
    java.util.Arrays.fill(features, feature);
    return gradient(index, y, parameter, features, prediction, lambda);
  }

  @Override
  public double[] gradient(int index, double y, Vector[] parameter, Vector[] feature, double[] prediction, double lambda) {
    double[] gradient = new double[1];
    double[] cumulativeProbs = cumulativeProbs(prediction);
    int c = (int) y;
    double cumulativeCMinus1 = c==0?0:cumulativeProbs[c-1];

    gradient[0] = (1/prediction[c]) * ((cumulativeProbs[c] * (1-cumulativeProbs[c]) * (-1)*feature[0].getQuick(index)) - (cumulativeCMinus1 * (1 - cumulativeCMinus1) * (-1)*feature[0].getQuick(index)));
    gradient[0] = index==0? learningRate * gradient[0] : learningRate * (gradient[0] - lambda * parameter[0].getQuick(index));
    return gradient;
  }

  @Override
  public double[] cutsGradient(double y, double[] cuts, double[] linearCombination, double lambda) {
    int c = (int)y;
    double probc;
    double[] cumulativeProbs;
    double[] gradient = new double[cuts.length];
    java.util.Arrays.fill(gradient, 0);

    cumulativeProbs = predictCumulative(linearCombination);
    probc = predictSingle(linearCombination)[c];

    double cumulativeCMinus1 = c==0?0:cumulativeProbs[c-1];

    for(int i = 0; i<c; i++){
      gradient[i] = 1/probc * (cumulativeProbs[c]*(1-cumulativeProbs[c])*1 - cumulativeCMinus1*(1-cumulativeCMinus1)*1);
    }
    if(c!=linearCombination.length){
      gradient[c] = 1/probc * (cumulativeProbs[c]*(1-cumulativeProbs[c])*1 - 0);
    }

    for(int i = 0; i<gradient.length; i++){
      gradient[i] = learningRate * (gradient[i] - lambda * cuts[i]);
    }
    return gradient;

  }

  public double[] cumulativeProbs(double[] prediction){
    double[] cumulativeProbs = new double[prediction.length];
    cumulativeProbs[0] = prediction[0];
    for(int i = 1; i<cumulativeProbs.length; i++){
      cumulativeProbs[i] = cumulativeProbs[i-1]+prediction[i];
    }
    return cumulativeProbs;
  }

  protected double[] predictCumulative(double[] linearCombination){
    return cumulativeProbs(hypothesis.predict(linearCombination));
  }

  private double[] predictSingle (double[] linearCombination){
    return hypothesis.predict(linearCombination);
  }

  @Override
  public void setLambda(double lambda) {
    //To change body of implemented methods use File | Settings | File Templates.
  }
}
