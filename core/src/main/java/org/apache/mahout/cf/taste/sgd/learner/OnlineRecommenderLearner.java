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

package org.apache.mahout.cf.taste.sgd.learner;

import org.apache.mahout.cf.taste.sgd.gradient.StochasticGradient;
import org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis;
import org.apache.mahout.cf.taste.sgd.model.FeatureVectorModel;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

/**
 * Base for SGD based recommendation learner.
 */
public abstract class OnlineRecommenderLearner {
  protected StochasticGradient gradient;
  protected Hypothesis hypothesis;
  protected FeatureVectorModel featureVectorModel;

  protected OnlineRecommenderLearner(StochasticGradient gradient, Hypothesis hypothesis, FeatureVectorModel featureVectorModel) {
    this.gradient = gradient;
    this.hypothesis = hypothesis;
    this.featureVectorModel = featureVectorModel;
  }

  /**
   * @return underlying {@link Hypothesis}
   */
  public Hypothesis getHypothesis(){
    return this.hypothesis;
  }

  /**
   * @return underlying {@link StochasticGradient}
   */
  public StochasticGradient getGradient() {
    return gradient;
  }

  /**
   * @return underlying {@link FeatureVectorModel}
   */
  public FeatureVectorModel getFeatureVectorModel() {
    return featureVectorModel;
  }

  /**
   * see {@link FeatureVectorModel#initializeUserIfNeeded(long)}
   */
  public void initializeUserIfNeeded(long userId){
    featureVectorModel.initializeUserIfNeeded(userId);
  }

  /**
   * see {@link FeatureVectorModel#initializeItemIfNeeded(long)}
   */
  public void initializeItemIfNeeded(long itemId){
    featureVectorModel.initializeItemIfNeeded(itemId);
  }

  /**
   * trains the recommender with one example
   * @param first user id
   * @param second item id
   * @param y preference
   */
  public abstract void train(long first, long second, double y);
  public abstract void commit();

  /**
   * computes the linear combination and returns the prediction
   * @param first user id
   * @param second item id
   * @return see {@link Hypothesis#predict(double[])}
   */
  public abstract double[] predict(long first, long second);

  /**
   * computes the linear combination and returns the prediction
   * @param first user id
   * @param second item id
   * @return see {@link Hypothesis#predictFull(double[])}
   */
  public abstract double[] predictFull(long first, long second);



  public double[] getUpdatedCuts(double y, double[] cuts, double[] linearCombination, double cuts_lambda) {
    double[] updatedCuts = new double[cuts.length];
    double[] updateTerms = gradient.cutsGradient(y, cuts, linearCombination, cuts_lambda);
    for(int i = 0; i<updatedCuts.length; i++){
      updatedCuts[i] = cuts[i]+updateTerms[i];
    }
    return updatedCuts;
  }

  protected Vector[] getUpdatedTerms(Vector[] paramVector, Vector[] featureVector, double y, double[] prediction, double lambda, Map<Integer, Double> otherLambdaIfNecessary){
    return getUpdatedTerms(paramVector, featureVector, y, prediction, -1, lambda, otherLambdaIfNecessary);
  }

  protected Vector[] getUpdatedTerms(Vector[] paramVector, Vector featureVector, double y, double[] prediction, double lambda, Map<Integer, Double> otherLambdaIfNecessary){
    return getUpdatedTerms(paramVector, featureVector, y, prediction, -1, lambda, otherLambdaIfNecessary);
  }

  protected Vector[] getUpdatedTerms(Vector[] paramVector, Vector featureVector, double y, double[] prediction, int except, double lambda, Map<Integer, Double> otherLambdaIfNecessary){
    Vector[] updatedParams = new Vector[paramVector.length];

    Iterator<Vector.Element> it = featureVector.iterateNonZero();

    for(int c = 0; c<paramVector.length; c++){
      updatedParams[c] = paramVector[c].like();
    }

    while(it.hasNext()){
      int i = it.next().index();

      double[] updateTerms;

      if(i==except){
        updateTerms = new double[paramVector.length];
        for(int t = 0; t<updateTerms.length; t++){
          updateTerms[t] = 0;
        }
      }
      else{
        if(otherLambdaIfNecessary!=null&&otherLambdaIfNecessary.containsKey(i)){
          updateTerms = gradient.gradient(i, y, paramVector, featureVector, prediction, otherLambdaIfNecessary.get(i));
        }
        else{
          updateTerms = gradient.gradient(i, y, paramVector, featureVector, prediction, lambda);
        }
      }
      for(int c = 0; c<updateTerms.length; c++){
        double d = updateTerms[c];
        updatedParams[c].setQuick(i, paramVector[c].getQuick(i) + d);
      }
    }

    return updatedParams;
  }
  protected Vector[] getUpdatedTerms(Vector[] paramVector, Vector[] featureVector, double y, double[] prediction, int except, double lambda, Map<Integer, Double> otherLambdaIfNecessary){
    Vector[] updatedParams = new Vector[paramVector.length];

    Iterator<Vector.Element> it = featureVector[0].iterateNonZero();

    for(int c = 0; c<paramVector.length; c++){
      updatedParams[c] = paramVector[c].like();
    }

    while(it.hasNext()){
      int i = it.next().index();
      double[] updateTerms;
      if(i==except){
        updateTerms = new double[paramVector.length];
        for(int t = 0; t<updateTerms.length; t++){
          updateTerms[t] = 0;
        }
      }
      else{
        if(otherLambdaIfNecessary!=null && otherLambdaIfNecessary.containsKey(i)){
          updateTerms = gradient.gradient(i, y, paramVector, featureVector, prediction, otherLambdaIfNecessary.get(i));
        }
        else{
          updateTerms = gradient.gradient(i, y, paramVector, featureVector, prediction, lambda);
        }
      }
      for(int c = 0; c<updateTerms.length; c++){
        double d = updateTerms[c];
        updatedParams[c].setQuick(i, paramVector[c].getQuick(i) + d);
      }
    }

    return updatedParams;

  }


  protected static ArrayList<Integer> getNonZeroIndices(Vector v){
    ArrayList<Integer> indices = new ArrayList<Integer>();
    Iterator<Vector.Element> it = v.iterateNonZero();
    while(it.hasNext()){
      indices.add(it.next().index());
    }
    return indices;
  }
}
