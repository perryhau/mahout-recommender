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

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.sgd.gradient.StochasticGradient;
import org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis;
import org.apache.mahout.cf.taste.sgd.model.FeatureVectorModel;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * A factor based, side info aware {@link OnlineRecommenderLearner}
 */
public class SideInfoAwareRecommenderLearner extends OnlineRecommenderLearner {

  private double factorLambda, onUserSideLambda, onItemSideLambda, onDynamicSideLambda;
  private double cutsLambda = 0;
  private Map<Integer, Double> biasLambdas = new HashMap<Integer, Double>();
  private  static Logger logger = LoggerFactory.getLogger(SideInfoAwareRecommenderLearner.class);

  /**
   * @param gradient underlying gradient
   * @param hypothesis underlying hypothesis
   * @param featureVectorModel underlying featureVectorModel
   * @param biasLambda regularization rate to be passed to underlying {@link StochasticGradient} for user and item biases
   * @param factorLambda regularization rate to be passed to underlying {@link StochasticGradient} for user and item factors
   * @param onUserSideLambda regularization rate to be passed to underlying {@link StochasticGradient} for user side info
   * @param onItemSideLambda regularization rate to be passed to underlying {@link StochasticGradient} for item side info
   * @param onDynamicSideLambda regularization rate to be passed to underlying {@link StochasticGradient} for dynamic side info
   */
  public SideInfoAwareRecommenderLearner(StochasticGradient gradient, Hypothesis hypothesis, FeatureVectorModel featureVectorModel, double biasLambda, double factorLambda, double onUserSideLambda, double onItemSideLambda, double onDynamicSideLambda) {
    super(gradient, hypothesis, featureVectorModel);
    biasLambdas.put(1, biasLambda);
    biasLambdas.put(2, biasLambda);
    this.factorLambda = factorLambda;
    this.onUserSideLambda = onUserSideLambda;
    this.onItemSideLambda = onItemSideLambda;
    this.onDynamicSideLambda = onDynamicSideLambda;
    logger.info("Constructing a factor model with side info");
    logger.info("Lambda for factors: "+ factorLambda);
    logger.info("Lambda for bias terms: "+biasLambda);
    logger.info("Lambda for user side info parameters: "+ onUserSideLambda);
    logger.info("Lambda for item side info parameters: "+ onItemSideLambda);
    logger.info("Lambda for dynamic side info parameters: "+ onDynamicSideLambda);
  }

  /**
   * @param gradient underlying gradient
   * @param hypothesis underlying hypothesis
   * @param featureVectorModel underlying featureVectorModel
   * @param biasLambda regularization rate to be passed to underlying {@link StochasticGradient} for user and item biases
   * @param factorLambda regularization rate to be passed to underlying {@link StochasticGradient} for user and item factors
   * @param onUserSideLambda regularization rate to be passed to underlying {@link StochasticGradient} for user side info
   * @param onItemSideLambda regularization rate to be passed to underlying {@link StochasticGradient} for item side info
   * @param onDynamicSideLambda regularization rate to be passed to underlying {@link StochasticGradient} for dynamic side info
   * @param cutsLambda regularization rate to be passed to underlying {@link StochasticGradient} for ordinal cuts
   */
  public SideInfoAwareRecommenderLearner(StochasticGradient gradient, Hypothesis hypothesis, FeatureVectorModel featureVectorModel, double biasLambda, double factorLambda, double onUserSideLambda, double onItemSideLambda, double onDynamicSideLambda, double cutsLambda) {
    this(gradient, hypothesis, featureVectorModel, biasLambda, factorLambda, onUserSideLambda, onItemSideLambda, onDynamicSideLambda);
    this.cutsLambda = cutsLambda;
    logger.info("Lambda for ordinal cuts: "+cutsLambda);
  }

  @Override
  public void train(long first, long second, double y) {
    Vector[] alphas = featureVectorModel.getAlphas(first);
    Vector[] betas = featureVectorModel.getBetas(second);
    Vector[] thetasOnTs = featureVectorModel.getThetasOnTs(first);
    Vector[] thetasOnZs = featureVectorModel.getThetasOnZs(first);
    Vector[] gammasOnXs = featureVectorModel.getGammasOnXs(second);
    Vector[] gammasOnZs = featureVectorModel.getGammasOnZs(second);
    double[] cuts = featureVectorModel.getCuts();

    Vector x = featureVectorModel.getXs(first);
    Vector t = featureVectorModel.getTs(second);
    Vector z = featureVectorModel.getZs(first, second);

    double[] linearCombination = linearCombination(cuts, alphas, betas, thetasOnTs, thetasOnZs, gammasOnXs, gammasOnZs, x, t, z);
    double[] prediction = hypothesis.predict(linearCombination);

    double[] cutsNew = getUpdatedCuts(y, cuts,  linearCombination, cutsLambda);
    Vector[] alphasNew = getUpdatedTerms(alphas, betas, y, prediction,1, factorLambda, biasLambdas);
    Vector[] betasNew = getUpdatedTerms(betas, alphas, y, prediction,2, factorLambda, biasLambdas);
    Vector[] thetasOnTNew = getUpdatedTerms(thetasOnTs, t, y, prediction, onItemSideLambda, null);
    Vector[] thetasOnZNew = getUpdatedTerms(thetasOnZs, z, y, prediction, onDynamicSideLambda, null);
    Vector[] gammasOnXNew = getUpdatedTerms(gammasOnXs, x, y, prediction, onUserSideLambda, null);
    Vector[] gammasOnZNew = getUpdatedTerms(gammasOnZs, z, y, prediction, onDynamicSideLambda, null);

    featureVectorModel.setCuts(cutsNew);
    featureVectorModel.setAlphas(first, alphasNew);
    featureVectorModel.setBetas(second, betasNew);
    featureVectorModel.setThetasOnTs(first, thetasOnTNew);
    featureVectorModel.setThetasOnZs(first, thetasOnZNew);
    featureVectorModel.setGammasOnXs(second, gammasOnXNew);
    featureVectorModel.setGammasOnZs(second, gammasOnZNew);
  }

  @Override
  public double[] predict(long first, long second) {
    double[] cuts = featureVectorModel.getCuts();
    Vector[] alphas = featureVectorModel.getAlphas(first);
    Vector[] betas = featureVectorModel.getBetas(second);
    Vector[] thetasOnTs = featureVectorModel.getThetasOnTs(first);
    Vector[] thetasOnZs = featureVectorModel.getThetasOnZs(first);
    Vector[] gammasOnXs = featureVectorModel.getGammasOnXs(second);
    Vector[] gammasOnZs = featureVectorModel.getGammasOnZs(second);
    Vector x = featureVectorModel.getXs(first);
    Vector t = featureVectorModel.getTs(second);
    Vector z = featureVectorModel.getZs(first, second);
    return predict(cuts, alphas, betas, thetasOnTs, thetasOnZs, gammasOnXs, gammasOnZs, x, t, z);
  }

  @Override
  public double[] predictFull(long first, long second) {
    double[] cuts = featureVectorModel.getCuts();
    Vector[] alphas = featureVectorModel.getAlphas(first);
    Vector[] betas = featureVectorModel.getBetas(second);
    Vector[] thetasOnTs = featureVectorModel.getThetasOnTs(first);
    Vector[] thetasOnZs = featureVectorModel.getThetasOnZs(first);
    Vector[] gammasOnXs = featureVectorModel.getGammasOnXs(second);
    Vector[] gammasOnZs = featureVectorModel.getGammasOnZs(second);
    Vector x = featureVectorModel.getXs(first);
    Vector t = featureVectorModel.getTs(second);
    Vector z = featureVectorModel.getZs(first, second);
    return hypothesis.predictFull(linearCombination(cuts, alphas, betas, thetasOnTs, thetasOnZs, gammasOnXs, gammasOnZs, x, t, z));  //To change body of implemented methods use File | Settings | File Templates.
  }

  private double[] predict(double[] cuts, Vector[] alphas, Vector[] betas, Vector[] thetasOnTs, Vector[] thetasOnZs, Vector[] gammasOnXs, Vector[] gammasOnZs, Vector x, Vector t, Vector z) {
    Pair<Vector[], Vector[]> factors = Pair.of(alphas, betas);
    List<Pair<Vector[], Vector>> side = Lists.newArrayList(Pair.of(thetasOnTs, t), Pair.of(thetasOnZs, z), Pair.of(gammasOnXs, x), Pair.of(gammasOnZs, z));
    double[] linearCombination = hypothesis.linearCombination(cuts, factors, side);
    return hypothesis.predict(linearCombination);
  }

  private double[] linearCombination(double[] cuts, Vector[] alphas, Vector[] betas, Vector[] thetasOnTs, Vector[] thetasOnZs, Vector[] gammasOnXs, Vector[] gammasOnZs, Vector x, Vector t, Vector z) {
    Pair<Vector[], Vector[]> factors = Pair.of(alphas, betas);
    List<Pair<Vector[], Vector>> side = Lists.newArrayList(Pair.of(thetasOnTs, t), Pair.of(thetasOnZs, z), Pair.of(gammasOnXs, x), Pair.of(gammasOnZs, z));
    return hypothesis.linearCombination(cuts, factors, side);
  }

  @Override
  public void commit() {
    //To change body of implemented methods use File | Settings | File Templates.
  }

}
