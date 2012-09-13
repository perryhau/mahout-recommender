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
 * Gradient implementation that works for numerical, binary, and poisson recommendation
 */
public class RegularizedDefaultGradient extends StochasticGradient {
  private Hypothesis hypothesis;
  private double learningRate;
  Logger logger = LoggerFactory.getLogger(RegularizedDefaultGradient.class);

  @Override
  public void setLambda(double lambda) {
  }

  public RegularizedDefaultGradient(Hypothesis h, double learningRate) {
    this.hypothesis = Preconditions.checkNotNull(h);
    this.learningRate = learningRate;
    logger.info("Using default gradient with learning rate: "+learningRate);
  }

  public double[] gradient(int index, double y, Vector[] alphaI, Vector[] betaJ, double lambda) {
    Preconditions.checkArgument(Preconditions.checkNotNull(alphaI).length == 1, "Check the algorithm, there are more than one targets");
    double h = hypothesis.predict(alphaI, betaJ)[0];
    if (index == 0){
      return new double[]{learningRate * (y - h) };
    }
    return new double[]{learningRate * (betaJ[0].getQuick(index) * (y - h) - lambda * alphaI[0].getQuick(index))};
  }


  public double[] gradient(int index, double y, Vector[] parameter, Vector feature, double[] prediction, double lambda){
    Preconditions.checkNotNull(parameter);
    Preconditions.checkNotNull(feature);
    Preconditions.checkNotNull(prediction);

    Vector[] featureArray = new Vector[parameter.length];
    for(int i = 0; i<featureArray.length; i++ ){
      featureArray[i] = feature;
    }

    return gradient(index, y, parameter, featureArray, prediction, lambda);
  }

  public double[] gradient(int index, double y, Vector[] parameter, Vector[] feature, double[] prediction, double lambda){
    Preconditions.checkArgument(Preconditions.checkNotNull(parameter).length == Preconditions.checkNotNull(parameter).length);

    double h = prediction[0];
    if(index == 0){
      return new double[]{learningRate *  (y - h)};
    }
    return new double[]{learningRate * (feature[0].getQuick(index) * (y - h) - lambda * parameter[0].getQuick(index))};
  }

}
