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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * A factor based {@link OnlineRecommenderLearner}
 */
public class JustRatingBasedRecommenderLearner extends OnlineRecommenderLearner{
  private double factorLambda;
  private double cutsLambda = 0;
  private Map<Integer, Double> biasLambdas = new HashMap<Integer, Double>();
  private static Logger logger = LoggerFactory.getLogger(JustRatingBasedRecommenderLearner.class);

  /**
   * @param gradient underlying gradient
   * @param hypothesis underlying hypothesis
   * @param featureVectorModel underlying featureVectorModel
   * @param biasLambda regularization rate to be passed to underlying {@link StochasticGradient} for user and item biases
   * @param factorLambda regularization rate to be passed to underlying {@link StochasticGradient} for user and item factors
   */
  public JustRatingBasedRecommenderLearner(StochasticGradient gradient, Hypothesis hypothesis, FeatureVectorModel featureVectorModel, double biasLambda, double factorLambda) {
    super(gradient, hypothesis, featureVectorModel);
    this.factorLambda = factorLambda;
    biasLambdas.put(1, biasLambda);
    biasLambdas.put(2, biasLambda);
    logger.info("Constructing a factor model without side info");
    logger.info("Lambda for factors: "+ factorLambda);
    logger.info("Lambda for bias terms: "+biasLambda);
  }

  /**
   * @param gradient underlying gradient
   * @param hypothesis underlying hypothesis
   * @param featureVectorModel underlying featureVectorModel
   * @param bias_lambda regularization rate to be passed to underlying {@link StochasticGradient} for user and item biases
   * @param factorLambda regularization rate to be passed to underlying {@link StochasticGradient} for user and item factors
   * @param cutsLambda regularization rate to be passed to underlying {@link StochasticGradient} for ordinal cuts
   */
  public JustRatingBasedRecommenderLearner(StochasticGradient gradient, Hypothesis hypothesis, FeatureVectorModel featureVectorModel, double bias_lambda, double factorLambda, double cutsLambda) {
    this(gradient, hypothesis, featureVectorModel, bias_lambda, factorLambda);
    this.cutsLambda = cutsLambda;
    logger.info("Lambda for ordinal cuts: "+ cutsLambda);
  }

  @Override
  public void train(long first, long second, double y) {
    double[] cuts = featureVectorModel.getCuts();
    Vector[] alphas = featureVectorModel.getAlphas(first);
    Vector[] betas = featureVectorModel.getBetas(second);

    double[] linearCombination = hypothesis.linearCombination(cuts, alphas, betas);
    double[] prediction = predict(linearCombination);

    double[] cutsNew = getUpdatedCuts(y, cuts, linearCombination, cutsLambda);
    //update alphas using betas as features
    Vector[] alphasNew =  getUpdatedTerms(alphas, betas, y, prediction,1, factorLambda, biasLambdas);

    //update betas using alphas as features
    Vector[] betasNew = getUpdatedTerms(betas, alphas, y, prediction,2, factorLambda, biasLambdas);

    featureVectorModel.setCuts(cutsNew);
    featureVectorModel.setAlphas(first, alphasNew);
    featureVectorModel.setBetas(second, betasNew);
  }

  @Override
  public double[] predict(long first, long second) {
    double[] cuts = featureVectorModel.getCuts();
    Vector[] alphas = featureVectorModel.getAlphas(first);
    Vector[] betas = featureVectorModel.getBetas(second);
    return hypothesis.predict(hypothesis.linearCombination(cuts, alphas, betas));
  }

  @Override
  public double[] predictFull(long first, long second){
    double[] cuts = featureVectorModel.getCuts();
    Vector[] alphas = featureVectorModel.getAlphas(first);
    Vector[] betas = featureVectorModel.getBetas(second);
    return hypothesis.predictFull(hypothesis.linearCombination(cuts, alphas, betas));
  }

  private double[] predict(double[] linearCombination){
    return hypothesis.predict(linearCombination);
  }



  @Override
  public void commit() {
    //To change body of implemented methods use File | Settings | File Templates.
  }

}
