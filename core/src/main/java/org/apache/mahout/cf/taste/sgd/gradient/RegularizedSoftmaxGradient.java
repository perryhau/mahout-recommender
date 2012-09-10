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

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Gradient implementation for Multinomial Recommendation
 */

public class RegularizedSoftmaxGradient extends StochasticGradient {
  private Hypothesis hypothesis;
  private double learningRate;
  Logger logger = LoggerFactory.getLogger(RegularizedSoftmaxGradient.class);

  public RegularizedSoftmaxGradient(Hypothesis h, double learningRate) {
    this.hypothesis = h;
    this.learningRate = learningRate;
    logger.info("Using softmax gradient with learning rate: "+learningRate);

  }

  @Override
  public void setLambda(double lambda) {
  }


  public double[] gradient(int index, double y, Vector[] alphaI, Vector[] betaJ, double lambda) {
    Preconditions.checkArgument(alphaI.length > 1, "Check the algorithm, there should more than one feature vectors, one for each class");

    double [] probs = hypothesis.predict(alphaI, betaJ);
    double[] result = new double[probs.length];

    for (int c = 0; c<result.length; c++){
      if(index==0){
        result[c] = learningRate * (betaJ[c].getQuick(index) * indicator(c, (int)y)- probs[c]);
      }
      else{
        result[c] = learningRate * (betaJ[c].getQuick(index) * indicator(c, (int)y) - probs[c]- lambda * alphaI[c].getQuick(index)) ;
      }
    }
    return result;

  }

  public double[] gradient(int index, double y, Vector[] parameter, Vector feature, double[] prediction, double lambda){
    Vector[] featureForAllClasses = new Vector[parameter.length];
    java.util.Arrays.fill(featureForAllClasses, feature);
    return gradient(index, y, parameter, featureForAllClasses, prediction, lambda);
  }

  public double[] gradient(int index, double y, Vector[] parameter, Vector[] feature, double[] prediction, double lambda){
    Preconditions.checkNotNull(parameter);
    Preconditions.checkNotNull(feature);
    Preconditions.checkArgument(parameter.length == feature.length);

    double [] gradient = new double[parameter.length];

    for (int c = 0; c<gradient.length; c++){
      if(index==0){
        gradient[c] = learningRate * (feature[c].getQuick(index) * indicator(c, (int)y)- prediction[c]);
      }
      else{
        gradient[c] = learningRate * (feature[c].getQuick(index) * (indicator(c, (int)y) - prediction[c]) - lambda*parameter[c].getQuick(index));
      }
    }
    return gradient;

  }

  public static int indicator(int i, int y){
    return (i!=y)?0:1;
  }


}
