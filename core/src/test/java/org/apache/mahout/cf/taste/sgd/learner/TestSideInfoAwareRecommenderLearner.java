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

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.sgd.gradient.StochasticGradient;
import org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis;
import org.apache.mahout.cf.taste.sgd.model.FeatureVectorModel;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;
import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;
import java.util.Map;

import static org.easymock.EasyMock.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;


public class TestSideInfoAwareRecommenderLearner {
  Hypothesis hypothesis;
  StochasticGradient gradient;
  FeatureVectorModel featureVectorModel;
  SideInfoAwareRecommenderLearner learner;

  Vector alpha, beta, thetaOnT, thetaOnZ, gammaOnX, gammaOnZ, x, t, z;

  Vector[] alphas, betas, thetaOnTs, thetaOnZs, gammaOnXs, gammaOnZs;
  Vector[] alphasNew, betasNew, thetaOnTsNew, thetaOnZsNew, gammaOnXsNew, gammaOnZsNew;
  Vector[] multiAlphas = new Vector[3];
  Vector[] multiAlphasNew = new Vector[3];
  Vector[] multiBetas = new Vector[3];
  Vector[] multiBetasNew = new Vector[3];
  Vector[] multiThetaOnTs = new Vector[3];
  Vector[] multiThetaOnTsNew = new Vector[3];
  Vector[] multiThetaOnZs = new Vector[3];
  Vector[] multiThetaOnZsNew = new Vector[3];
  Vector[] multiGammaOnXs = new Vector[3];
  Vector[] multiGammaOnXsNew = new Vector[3];
  Vector[] multiGammaOnZs = new Vector[3];
  Vector[] multiGammaOnZsNew = new Vector[3];

  double[] cuts = new double[3];

  @Before
  public void setUp (){
    class SparseIncFunc implements DoubleFunction {
      @Override
      public double apply(double arg1) {
        if (arg1==0){
          return 0;
        }
        else return arg1+1;
      }
    }

    for(int i = 0; i<cuts.length; i++){
      cuts[i] = 0;
    }

    gradient = createMock(StochasticGradient.class);
    hypothesis = createMock(Hypothesis.class);
    featureVectorModel = createMock(FeatureVectorModel.class);

    alpha = new DenseVector(2);
    alpha.setQuick(0, 1);
    alpha.setQuick(1, 2);
    alphas = new Vector[]{alpha};
    alphasNew = new Vector[]{alpha.clone().assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return arg1+1;
      }
    })};
    multiAlphas[0] = alpha;
    for(int i = 1; i<multiAlphas.length; i++){
      multiAlphas[i] = multiAlphas[i-1].clone().assign(new DoubleFunction() {
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

    beta = new DenseVector(2);
    beta.setQuick(0,2);
    beta.setQuick(1,1);
    betas = new Vector[]{beta};
    betasNew = new Vector[]{beta.clone().assign(new DoubleFunction() {
      @Override
      public double apply(double arg1) {
        return arg1+1;
      }
    })};
    multiBetas[0] = beta;
    for(int i = 1; i<multiBetas.length; i++){
      multiBetas[i] = multiBetas[i-1].clone().assign(new DoubleFunction() {
        @Override
        public double apply(double arg1) {
          return arg1+1 ;
        }
      });
    }
    for(int i = 0; i<multiBetasNew.length; i++){
      multiBetasNew[i] = multiBetas[i].clone().assign(new DoubleFunction() {
        @Override
        public double apply(double arg1) {
          return arg1+1;
        }
      });
    }

    thetaOnT = new RandomAccessSparseVector(3);
    thetaOnT.setQuick(5, 1);
    thetaOnT.setQuick(2, 2);
    thetaOnT.setQuick(8, 3);
    thetaOnTs = new Vector[] {thetaOnT};
    thetaOnTsNew = new Vector[] {thetaOnT.clone().assign(new SparseIncFunc())};

    multiThetaOnTs[0] = thetaOnT;
    for(int i = 1; i<multiThetaOnTs.length; i++){
      multiThetaOnTs[i] = multiThetaOnTs[i-1].clone().assign(new SparseIncFunc());
    }
    for(int i = 0; i<multiThetaOnTsNew.length; i++){
      multiThetaOnTsNew[i] = multiThetaOnTs[i].clone().assign(new SparseIncFunc());
    }

    thetaOnZ = new RandomAccessSparseVector(2);
    thetaOnZ.setQuick(12, 2);
    thetaOnZ.setQuick(1, 3);
    thetaOnZs = new Vector[] {thetaOnZ};
    thetaOnZsNew = new Vector[] {thetaOnZ.clone().assign(new SparseIncFunc())};
    multiThetaOnZs[0] = thetaOnZ;
    for(int i = 1; i<multiThetaOnZs.length; i++){
      multiThetaOnZs[i] = multiThetaOnZs[i-1].clone().assign(new SparseIncFunc());
    }
    for(int i = 0; i<multiThetaOnZsNew.length; i++){
      multiThetaOnZsNew[i] = multiThetaOnZs[i].clone().assign(new SparseIncFunc());
    }

    gammaOnZ = new RandomAccessSparseVector(2);
    gammaOnZ.setQuick(12, 3);
    gammaOnZ.setQuick(1, 2);
    gammaOnZs = new Vector[] {gammaOnZ};
    gammaOnZsNew = new Vector[] {gammaOnZ.clone().assign(new SparseIncFunc())};
    multiGammaOnZs[0] = gammaOnZ;
    for(int i = 1; i<multiGammaOnZs.length; i++){
      multiGammaOnZs[i] = multiGammaOnZs[i-1].clone().assign(new SparseIncFunc());
    }
    for(int i = 0; i<multiGammaOnZsNew.length; i++){
      multiGammaOnZsNew[i] = multiGammaOnZs[i].clone().assign(new SparseIncFunc());
    }

    gammaOnX = new RandomAccessSparseVector(1);
    gammaOnX.setQuick(12, 2);
    gammaOnXs = new Vector[] {gammaOnX};
    gammaOnXsNew = new Vector[] {gammaOnX.clone().assign(new SparseIncFunc())};
    multiGammaOnXs[0] = gammaOnX;
    for(int i = 1; i<multiGammaOnXs.length; i++){
      multiGammaOnXs[i] = multiGammaOnXs[i-1].clone().assign(new SparseIncFunc());
    }
    for(int i = 0; i<multiGammaOnXsNew.length; i++){
      multiGammaOnXsNew[i] = multiGammaOnXs[i].clone().assign(new SparseIncFunc());
    }

    x = new RandomAccessSparseVector(1);
    x.setQuick(12, 10);
    z = new RandomAccessSparseVector(2);
    z.setQuick(12, 4);
    z.setQuick(1, 5);
    t = new RandomAccessSparseVector(3);
    t.setQuick(5, 3);
    t.setQuick(2, 2);
    t.setQuick(8, 1);

  }

  @Test
  public void singleClassTrain(){
    learner = new SideInfoAwareRecommenderLearner(gradient, hypothesis, featureVectorModel, 0, 0, 0, 0, 0){

      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector[] featureVector, double y, double[] prediction) {
        if(y!=1){
          assertTrue("y: " + y, false);
        }
        if(prediction[0]!=1){
          assertTrue("prediction: "+prediction[0], false);
        }
        if(paramVector[0].equals(alpha) && featureVector[0].equals(beta)){
          return alphasNew;
        }
        if(paramVector[0].equals(beta) && featureVector[0].equals(alpha)){
          return betasNew;
        }
        return null;
      }


      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector featureVector, double y, double[] prediction){
        if(paramVector[0].equals(thetaOnT) && featureVector.equals(t)){
          return thetaOnTsNew;
        }
        if(paramVector[0].equals(thetaOnZ) && featureVector.equals(z)){
          return thetaOnZsNew;
        }
        if(paramVector[0].equals(gammaOnX) && featureVector.equals(x)){
          return gammaOnXsNew;
        }
        if(paramVector[0].equals(gammaOnZ) && featureVector.equals(z)){
          return gammaOnZsNew;
        }
        return null;
      }

      @Override
      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector[] featureVector, double y, double[] prediction, int except, double lambda, Map<Integer, Double> otherLambdaIfNecessary) {
        return this.getUpdatedTerms(paramVector, featureVector, y, prediction);
      }

      @Override
      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector featureVector, double y, double[] prediction, int except, double lambda, Map<Integer, Double> otherLambdaIfNecessary) {
        return this.getUpdatedTerms(paramVector, featureVector, y, prediction);
      }

      @Override
      public double[] getUpdatedCuts(double y, double[] cuts, double[] linearCombination, double cuts_lambda) {
        return cuts;
      }
    };
    expect(hypothesis.linearCombination(cuts, Pair.of(alphas, betas), Lists.newArrayList(
        Pair.of(thetaOnTs, t), Pair.of(thetaOnZs, z), Pair.of(gammaOnXs, x), Pair.of(gammaOnZs, z)))).andReturn(new double[]{79});
    expect(hypothesis.predict(aryEq(new double[]{79}))).andReturn(new double[]{1});
    expect(featureVectorModel.getCuts()).andReturn(cuts);
    expect(featureVectorModel.getAlphas(1)).andReturn(alphas);
    expect(featureVectorModel.getBetas(2)).andReturn(betas);
    expect(featureVectorModel.getThetasOnTs(1)).andReturn(thetaOnTs);
    expect(featureVectorModel.getThetasOnZs(1)).andReturn(thetaOnZs);
    expect(featureVectorModel.getGammasOnXs(2)).andReturn(gammaOnXs);
    expect(featureVectorModel.getGammasOnZs(2)).andReturn(gammaOnZs);
    expect(featureVectorModel.getXs(1)).andReturn(x);
    expect(featureVectorModel.getTs(2)).andReturn(t);
    expect(featureVectorModel.getZs(1,2)).andReturn(z);

    featureVectorModel.setCuts(aryEq(cuts));
    featureVectorModel.setAlphas(1, alphasNew);
    featureVectorModel.setBetas(2, betasNew);
    featureVectorModel.setThetasOnTs(1, thetaOnTsNew);
    featureVectorModel.setThetasOnZs(1, thetaOnZsNew);
    featureVectorModel.setGammasOnXs(2, gammaOnXsNew);
    featureVectorModel.setGammasOnZs(2, gammaOnZsNew);
    replay(hypothesis, featureVectorModel);

    learner.train(1, 2, 1);

    verify(hypothesis, featureVectorModel);

  }

  @Test
  public void multiClassTrain(){
    learner = new SideInfoAwareRecommenderLearner(gradient, hypothesis, featureVectorModel,0,0,0,0,0){
      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector[] featureVector, double y, double[] prediction) {
        assertTrue("paramVector size: "+paramVector.length, paramVector.length == 3);
        if(y!=1){
          assertTrue("y: " + y, false);
        }
        if(prediction[0]!=1 && prediction[1]!=2 && prediction[3]!=3){
          assertTrue("prediction: "+prediction[0], false);
        }
        if(paramVector[1].equals(multiAlphas[1]) && featureVector[1].equals(multiBetas[1])){
          return multiAlphasNew;
        }
        if(paramVector[1].equals(multiBetas[1]) && featureVector[1].equals(multiAlphas[1])){
          return multiBetasNew;
        }
        return null;
      }

      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector featureVector, double y, double[] prediction){
        if(paramVector[1].equals(multiThetaOnTs[1]) && featureVector.equals(t)){
          return multiThetaOnTsNew;
        }
        if(paramVector[1].equals(multiThetaOnZs[1]) && featureVector.equals(z)){
          return multiThetaOnZsNew;
        }
        if(paramVector[1].equals(multiGammaOnXs[1]) && featureVector.equals(x)){
          return multiGammaOnXsNew;
        }
        if(paramVector[1].equals(multiGammaOnZs[1]) && featureVector.equals(z)){
          return multiGammaOnZsNew;
        }
        return null;
      }

      @Override
      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector[] featureVector, double y, double[] prediction, int except, double lambda, Map<Integer, Double> otherLambdaIfNecessary) {
        return this.getUpdatedTerms(paramVector, featureVector, y, prediction);
      }

      @Override
      public Vector[] getUpdatedTerms(Vector[] paramVector, Vector featureVector, double y, double[] prediction, int except, double lambda, Map<Integer, Double> otherLambdaIfNecessary) {
        return this.getUpdatedTerms(paramVector, featureVector, y, prediction);
      }

      @Override
      public double[] getUpdatedCuts(double y, double[] cuts, double[] linearCombination, double cuts_lambda) {
        return cuts;
      }
    };


    expect(hypothesis.linearCombination(cuts, Pair.of(multiAlphas, multiBetas), Lists.newArrayList(
        Pair.of(multiThetaOnTs, t), Pair.of(multiThetaOnZs, z), Pair.of(multiGammaOnXs, x), Pair.of(multiGammaOnZs, z)))).andReturn(new double[]{79, 121, 167});

    expect(hypothesis.predict(aryEq(new double[]{79,121,167}))).andReturn(new double[]{1,2,3});
    expect(featureVectorModel.getCuts()).andReturn(cuts);
    expect(featureVectorModel.getAlphas(1)).andReturn(multiAlphas);
    expect(featureVectorModel.getBetas(2)).andReturn(multiBetas);
    expect(featureVectorModel.getThetasOnTs(1)).andReturn(multiThetaOnTs);
    expect(featureVectorModel.getThetasOnZs(1)).andReturn(multiThetaOnZs);
    expect(featureVectorModel.getGammasOnXs(2)).andReturn(multiGammaOnXs);
    expect(featureVectorModel.getGammasOnZs(2)).andReturn(multiGammaOnZs);
    expect(featureVectorModel.getXs(1)).andReturn(x);
    expect(featureVectorModel.getTs(2)).andReturn(t);
    expect(featureVectorModel.getZs(1,2)).andReturn(z);

    featureVectorModel.setCuts(cuts);
    featureVectorModel.setAlphas(1, multiAlphasNew);
    featureVectorModel.setBetas(2, multiBetasNew);
    featureVectorModel.setThetasOnTs(1, multiThetaOnTsNew);
    featureVectorModel.setThetasOnZs(1, multiThetaOnZsNew);
    featureVectorModel.setGammasOnXs(2, multiGammaOnXsNew);
    featureVectorModel.setGammasOnZs(2, multiGammaOnZsNew);
    replay(hypothesis, featureVectorModel);

    learner.train(1, 2, 1);

    verify(hypothesis, featureVectorModel);

  }

  @Test
  public void getUpdatedTermsSingleClass(){
    learner = new SideInfoAwareRecommenderLearner(gradient, hypothesis, featureVectorModel, 0, 0, 0, 0, 0);
    double[] predictionSingle = new double[]{100};
    double[] predictionMulti = new double[]{100,101,102};

    Iterator<Vector.Element> it;

    it = alpha.iterateNonZero();
    while(it.hasNext()){
      int i = it.next().index();
      expect(gradient.gradient(i, 1, alphas, betas, predictionSingle,0)).andReturn(new double[]{1});
    }

    it = gammaOnX.iterateNonZero();
    while(it.hasNext()){
      int i = it.next().index();
      expect(gradient.gradient(i, 1, gammaOnXs, x, predictionSingle, 0)).andReturn(new double[]{1});
    }
    it = gammaOnZ.iterateNonZero();
    while(it.hasNext()){
      int i = it.next().index();
      expect(gradient.gradient(i, 1, gammaOnZs, z, predictionSingle, 0)).andReturn(new double[]{1});
    }


    it = beta.iterateNonZero();
    while(it.hasNext()){
      int i = it.next().index();
      expect(gradient.gradient(i, 1, multiBetas, multiAlphas, predictionMulti, 0)).andReturn(new double[]{1, 1, 1});
    }
    it = thetaOnT.iterateNonZero();
    while(it.hasNext()){
      int i = it.next().index();
      expect(gradient.gradient(i, 1, multiThetaOnTs, t, predictionMulti, 0)).andReturn(new double[]{1, 1, 1});

    }
    it = thetaOnZ.iterateNonZero();
    while(it.hasNext()){
      int i = it.next().index();
      expect(gradient.gradient(i, 1, multiThetaOnZs, z, predictionMulti, 0)).andReturn(new double[]{1, 1, 1});
    }

    replay(gradient);
    Vector[] updatedAlphas = learner.getUpdatedTerms(alphas, betas, 1, predictionSingle, -1, 0, null);
    Vector[] updatedGammaOnXs = learner.getUpdatedTerms(gammaOnXs, x, 1, predictionSingle, -1, 0, null);
    Vector[] updatedGammaOnZs = learner.getUpdatedTerms(gammaOnZs, z, 1, predictionSingle, -1, 0, null);

    Vector[] updatedMultiBetas = learner.getUpdatedTerms(multiBetas, multiAlphas, 1, predictionMulti, -1, 0, null);
    Vector[] updatedMultiThetaOnTs = learner.getUpdatedTerms(multiThetaOnTs, t, 1, predictionMulti, -1, 0, null);
    Vector[] updatedMultiThetaOnZs = learner.getUpdatedTerms(multiThetaOnZs, z, 1, predictionMulti, -1, 0, null);

    verify(gradient);

    assertArrayEquals(alphasNew, updatedAlphas);
    assertArrayEquals(gammaOnXsNew, updatedGammaOnXs);
    assertArrayEquals(gammaOnZsNew, updatedGammaOnZs);

    assertArrayEquals(multiBetasNew, updatedMultiBetas);
    assertArrayEquals(multiThetaOnTsNew,updatedMultiThetaOnTs );
    assertArrayEquals(multiThetaOnZsNew,updatedMultiThetaOnZs );


  }


}
