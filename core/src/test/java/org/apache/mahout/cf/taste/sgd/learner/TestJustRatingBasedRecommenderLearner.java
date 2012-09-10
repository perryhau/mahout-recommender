/**
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

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.sgd.gradient.StochasticGradient;
import org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis;
import org.apache.mahout.cf.taste.sgd.model.FeatureVectorModel;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static org.easymock.EasyMock.*;

public class TestJustRatingBasedRecommenderLearner extends TasteTestCase{
  Hypothesis hypothesis;
  StochasticGradient gradient;
  FeatureVectorModel featureVectorModel;
  JustRatingBasedRecommenderLearner learner;

  Vector alpha, beta;
  Vector[] alphas, betas, alphasNew, betasNew;
  Vector[] multiAlphas = new Vector[3];
  Vector[] multiBetas = new Vector[3];
  Vector[] multiAlphasNew = new Vector[3];
  Vector[] multiBetasNew = new Vector[3];
  double[] cuts = new double[3];
  double[] globalBias;
  double[] multiGlobalBias;
  private Vector[] singleGlobalBiasVector;
  private Vector[] multiGlobalBiasVector;

  @Before
  public void setUp(){
    gradient = createMock(StochasticGradient.class);
    hypothesis = createMock(Hypothesis.class);
    featureVectorModel = createMock(FeatureVectorModel.class);
    for(int i = 0; i<cuts.length; i++){
      cuts[i] = 0;
    }
    globalBias = new double[]{0};
    singleGlobalBiasVector = new Vector[]{new DenseVector(globalBias)};
    multiGlobalBias = new double[]{0,0,0};
    multiGlobalBiasVector = new Vector[]{new DenseVector(new double[]{multiGlobalBias[0]}),new DenseVector(new double[]{multiGlobalBias[1]}),new DenseVector(new double[]{multiGlobalBias[1]})};
    alpha = new DenseVector(5);
    alpha.setQuick(0,3);
    alpha.setQuick(1,2);
    alpha.setQuick(2,1);
    alpha.setQuick(3,2);
    alpha.setQuick(4,3);
    beta = new DenseVector(5);
    beta.setQuick(0,1);
    beta.setQuick(1,2);
    beta.setQuick(2,3);
    beta.setQuick(3,2);
    beta.setQuick(4,1);

    alphas = new Vector[]{alpha};
    betas = new Vector[]{beta};

    alphasNew = new Vector[]{alpha.clone().assign(new DoubleFunction() {
      @Override
      public double apply(double v) {
        return v+1;
      }
    })};

    betasNew = new Vector[]{beta.clone().assign(new DoubleFunction() {
      @Override
      public double apply(double v) {
        return v+1;
      }
    })};
    multiAlphas[0] = alpha;
    multiAlphas[1] = alpha.clone().assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return arg1+1;
      }
    });
    multiAlphas[2] = alpha.clone().assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return arg1+2;
      }
    });
    multiBetas[0] = beta;
    multiBetas[1] = beta.clone().assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return arg1+1;
      }
    });
    multiBetas[2] = beta.clone().assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return arg1+2;
      }
    });
    for(int i = 0; i<multiBetasNew.length; i++){
      multiBetasNew[i] = multiBetas[i].clone().assign(new DoubleFunction() {
        @Override
        public double apply(double arg1) {
          return arg1+1;
        }
      });
    }
    for(int i = 0; i<multiAlphasNew.length; i++){
      multiAlphasNew[i] = multiAlphas[i].clone().assign(new DoubleFunction() {
        @Override
        public double apply(double arg1) {
          return arg1+1;
        }
      });
    }
  }

  @Test
  public void singleClassTrain(){
    learner = new JustRatingBasedRecommenderLearner(gradient, hypothesis, featureVectorModel, 0, 0){
      public Vector[] getUpdatedTerms(Vector[] firstFactors, Vector[] secondFactors, double y, double[] prediction) {
        if(firstFactors==null){
          return null;
        }
        if(firstFactors[0].equals(alphas[0]))
          return alphasNew;
        if(firstFactors[0].equals(betas[0]))
          return betasNew;

        return singleGlobalBiasVector;

      }

      @Override
      public double[] getUpdatedCuts(double y, double[] cuts, double[] linearCombination, double cuts_lambda) {
        return new double[]{0,0,0};
      }

      @Override
      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector[] featureVector, double y, double[] prediction, int except, double lambda, Map<Integer, Double> otherLambdaIfNecessary) {
        return this.getUpdatedTerms(paramVector, featureVector, y, prediction);
      }
    };

    expect(featureVectorModel.getCuts()).andReturn(cuts);
    expect(featureVectorModel.getAlphas(1)).andReturn(alphas);
    expect(featureVectorModel.getBetas(2)).andReturn(betas);
    expect(hypothesis.linearCombination(cuts, alphas, betas)).andReturn(new double[]{17});
    expect(hypothesis.predict(aryEq(new double[]{17}))).andReturn(null);

    featureVectorModel.setCuts(aryEq(cuts));
    featureVectorModel.setAlphas(1, alphasNew);
    featureVectorModel.setBetas(2, betasNew);

    replay(featureVectorModel, hypothesis, gradient);

    learner.train(1, 2, 1);

    verify(featureVectorModel, hypothesis, gradient);
  }

  @Test
  public void multiClassTrain(){
    learner = new JustRatingBasedRecommenderLearner(gradient, hypothesis, featureVectorModel, 0, 0){

      public Vector[] getUpdatedTerms(Vector[] firstFactors, Vector[] secondFactors, double y, double[] prediction) {
        if(y!=1){
          return null;
        }

        if(firstFactors[0].equals(multiAlphas[0])){
          return multiAlphasNew;
        }
        if(firstFactors[0].equals(multiBetas[0])){
          return multiBetasNew;
        }

        return multiGlobalBiasVector;
      }

      @Override
      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector[] featureVector, double y, double[] prediction, int except, double lambda, Map<Integer, Double> otherLambdaIfNecessary) {
        return this.getUpdatedTerms(paramVector, featureVector, y, prediction);
      }

      @Override
      public double[] getUpdatedCuts(double y, double[] cuts, double[] linearCombination, double cuts_lambda) {
        return new double[] {0,0,0};
      }
    };

    expect(featureVectorModel.getCuts()).andReturn(cuts);
    expect(featureVectorModel.getAlphas(1)).andReturn(multiAlphas);
    expect(featureVectorModel.getBetas(2)).andReturn(multiBetas);

    expect(hypothesis.linearCombination(cuts, multiAlphas, multiBetas)).andReturn(new double[]{17,42,77});
    expect(hypothesis.predict(aryEq(new double[]{17,42,77}))).andReturn(null);


    featureVectorModel.setCuts(aryEq(cuts));
    featureVectorModel.setAlphas(1, multiAlphasNew);
    featureVectorModel.setBetas(2, multiBetasNew);
    replay(hypothesis, featureVectorModel);

    learner.train(1,2,1);

    verify(hypothesis, featureVectorModel);

  }

  @Test
  public void getUpdatedTerms(){
    learner = new JustRatingBasedRecommenderLearner(gradient, hypothesis, featureVectorModel, 0, 0);

    for(int i = 0; i<5; i++){
      expect(gradient.gradient(i, 1, alphas, betas, null, 0)).andReturn(new double[]{1});
    }
    for(int i = 0; i<5; i++){
      expect(gradient.gradient(i, 1, betas, alphas, null, 0)).andReturn(new double[]{1});
    }
    for(int i = 0; i<5; i++){
      expect(gradient.gradient(i, 1, multiAlphas, multiBetas, null, 0)).andReturn(new double[]{1,1,1});
    }
    for(int i = 0; i<5; i++){
      expect(gradient.gradient(i, 1, multiBetas, multiAlphas, null, 0)).andReturn(new double[]{1,1,1});
    }

    replay( gradient);

    Vector[] updatedAlphas = learner.getUpdatedTerms(alphas, betas, 1, null, 0, null);
    Vector[] updatedBetas = learner.getUpdatedTerms(betas, alphas, 1, null, 0, null);

    Vector[] updatedMultiAlphas = learner.getUpdatedTerms(multiAlphas, multiBetas, 1, null, 0, null);
    Vector[] updatedMultiBetas = learner.getUpdatedTerms(multiBetas, multiAlphas, 1, null, 0, null);

    verify(gradient);

    for(int i = 0; i<alphas.length; i++){
      assertEquals(alphasNew[i], updatedAlphas[i]);
    }
    for(int i = 0; i<betas.length; i++){
      assertEquals(betasNew[i], updatedBetas[i]);
    }
    for(int i = 0; i<multiAlphas.length; i++){
      assertEquals(multiAlphasNew[i], updatedMultiAlphas[i]);
    }
    for(int i = 0; i<multiBetas.length; i++){
      assertEquals(multiBetasNew[i], updatedMultiBetas[i]);
    }
  }

  @Test
  public void testPredict(){
    learner = new JustRatingBasedRecommenderLearner(gradient, hypothesis, featureVectorModel, 0, 0);
    expect(featureVectorModel.getCuts()).andReturn(cuts);
    expect(featureVectorModel.getAlphas(1)).andReturn(alphas);
    expect(featureVectorModel.getBetas(2)).andReturn(betas);
    expect(hypothesis.linearCombination(cuts, alphas, betas)).andReturn(new double[]{17});
    expect(hypothesis.predict(aryEq(new double[]{17}))).andReturn(new double[]{18});
    replay(featureVectorModel, hypothesis);
    learner.predict(1,2);
  }
}
