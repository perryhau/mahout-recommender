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

package org.apache.mahout.cf.taste.sgd.experiment;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.impl.model.FactorizationAwareDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericIncrementalDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.sgd.common.MostProbableClassPredictionStrategy;
import org.apache.mahout.cf.taste.sgd.common.RatingMapper;
import org.apache.mahout.cf.taste.sgd.common.RatingPredictionStrategy;
import org.apache.mahout.cf.taste.sgd.common.ScoreOnTargetClassStrategy;
import org.apache.mahout.cf.taste.sgd.eval.Eval;
import org.apache.mahout.cf.taste.sgd.eval.RecommenderEvalSetup;
import org.apache.mahout.cf.taste.sgd.gradient.RegularizedDefaultGradient;
import org.apache.mahout.cf.taste.sgd.gradient.RegularizedOrdinalGradient;
import org.apache.mahout.cf.taste.sgd.gradient.RegularizedSoftmaxGradient;
import org.apache.mahout.cf.taste.sgd.gradient.StochasticGradient;
import org.apache.mahout.cf.taste.sgd.hypothesis.*;
import org.apache.mahout.cf.taste.sgd.learner.JustRatingBasedRecommenderLearner;
import org.apache.mahout.cf.taste.sgd.learner.OnlineRecommenderLearner;
import org.apache.mahout.cf.taste.sgd.learner.SideInfoAwareRecommenderLearner;
import org.apache.mahout.cf.taste.sgd.model.FeatureVectorModel;
import org.apache.mahout.cf.taste.sgd.model.InMemoryFeatureVectorModel;
import org.apache.mahout.cf.taste.sgd.recommender.OnlineFactorizationRecommender;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class ExperimentDriver extends AbstractJob{
  private static Logger logger = LoggerFactory.getLogger(ExperimentDriver.class);

  public static String ORDINAL = "ordinal";
  public static String SIDE_INFO = "sideInfo";
  public static String ITERATION = "iteration";
  public static String CLASSES = "classes";
  public static String BIAS_LAMBDA = "biasLambda";
  public static String FACTORS_LAMBDA = "factorsLambda";
  public static String USER_SIDE_LAMBDA = "userSideLambda";
  public static String ITEM_SIDE_LAMBDA = "itemSideLambda";
  public static String DYNAMIC_SIDE_LAMBDA = "dynamicSideLambda";
  public static String CUTS_LAMBDA = "cutsLambda";
  public static String FACTOR_SIZE = "factorSize";
  public static String LEARNING_RATE = "learningRate";
  public static String RATING_MAPPER = "ratingMapper";
  public static String EVAL = "eval";
  public static String USER_SIDE_INFO_FILE = "userSideInfoFile";
  public static String ITEM_SIDE_INFO_FILE = "itemSideInfoFile";
  public static String TRAINING_FILE = "trainingFile";
  public static String TEST_FILE = "testFile";
  public static String SEPARATOR = "separator";
  public static String SEE_CONVERGENCE = "seeConvergence";

  public ExperimentDriver() {
  }

  @Override
  public int run(String[] strings) throws Exception {
    addOption(ORDINAL, "o", "Set true if ordinal, default false", "false");
    addOption(SIDE_INFO, "s", "Set true for side info aware recommendation, default false", "false");
    addOption(ITERATION, "i", "Number of iterations, default 50", "50");
    addOption(CLASSES, "c", "Number of classes, set 2 if this is binary, or # of categories if this is multinomial or ordinal, default 1 for numerical recommendation", "1");
    addOption(BIAS_LAMBDA, "bl", "Regularization rate for user and item intercepts, default 0.005", "0.005");
    addOption(FACTORS_LAMBDA, "fl", "Regularization rate for latent factors, default 0.025", "0.025");
    addOption(USER_SIDE_LAMBDA, "ul", "Regularization rate for user side information, need to be set for user side info aware recommender", false);
    addOption(ITEM_SIDE_LAMBDA, "il", "Regularization rate for item side information, need to be set for item side info aware recommender", false);
    addOption(DYNAMIC_SIDE_LAMBDA, "dl", "Regularization rate for dynamic side information, need to be set for dynamic side info aware recommender", false);
    addOption(CUTS_LAMBDA, "cl", "Regularization rate for ordinal cuts, need to be set for ordinal recommender", false);
    addOption(FACTOR_SIZE, "fs", "Number of factors, default 150", "150");
    addOption(LEARNING_RATE, "lr", "Learning rate, default 0.005", "0.005");
    addOption(RATING_MAPPER, "mapper", "Rating mapper class name, default org.apache.mahout.cf.taste.sgd.common.DefaultRatingMapper", "org.apache.mahout.cf.taste.sgd.common.DefaultRatingMapper");
    addOption(EVAL, "e", "Evaluation class name", true);
    addOption(SEE_CONVERGENCE, "con", "If true, prints current score after each iteration", true);
    addOption(USER_SIDE_INFO_FILE, "userSide", "SequenceFile of <LongWritable, VectorWritable> pairs for user side info. Set this if this is a user side info aware recommender", false);
    addOption(ITEM_SIDE_INFO_FILE, "itemSide", "SequenceFile of <LongWritable, VectorWritable> pairs for item side info. Set this if this is a item side info aware recommender", false);
    addOption(TRAINING_FILE, "train", "Path for text file of training ratings <userID#itemID#rating#[remaining]> format is expected where # is the SEPARATOR", true);
    addOption(TEST_FILE, "test", "Path for text file of test ratings <userID#itemID#rating#[remaining]> format is expected where # is the SEPARATOR", true);
    addOption(SEPARATOR, "sep", "Separating string of userId, itemId and ratings in training and test file, default is COMMA character", ",");
    addOption(DefaultOptionCreator.helpOption());
    parseArguments(strings);

    boolean ordinal = getOption(ORDINAL).equalsIgnoreCase("true");
    boolean sideInfo = getOption(SIDE_INFO).equalsIgnoreCase("true");
    boolean seeConvergence = getOption(SEE_CONVERGENCE).equalsIgnoreCase("true");
    int numIterations = Integer.parseInt(getOption(ITERATION));
    int numClasses = Integer.parseInt(getOption(CLASSES));
    double biasLambda = Double.parseDouble(getOption(BIAS_LAMBDA));
    double factorsLambda = Double.parseDouble(getOption(FACTORS_LAMBDA));
    double onUserSideLambda = 0;
    double onItemSideLambda = 0;
    double onDynamicSideLambda = 0;
    double cutsLambda = 0;
    int factorSize = Integer.parseInt(getOption(FACTOR_SIZE));
    double learningRate = Double.parseDouble(getOption(LEARNING_RATE));
    Class<? extends RatingMapper> ratingMapperClass = Class.forName(getOption(RATING_MAPPER)).asSubclass(RatingMapper.class);
    RatingMapper ratingMapper = ratingMapperClass.newInstance();
    String trainingFilePath = getOption(TRAINING_FILE);
    String testFilePath = getOption(TEST_FILE);
    String separator = getOption(SEPARATOR);

    StochasticGradient gradient;
    Hypothesis hypothesis;
    OnlineFactorizationRecommender recommender;
    OnlineRecommenderLearner learner;
    DataModel dataModel;
    FeatureVectorModel featureVectorModel = new InMemoryFeatureVectorModel(factorSize, numClasses, ordinal);
    Eval eval;
    String userSideInfoFile, itemSideInfoFile;

    if (ordinal) {
      cutsLambda = Double.parseDouble(getOption(CUTS_LAMBDA));
      hypothesis = new OrdinalHypothesis();
      gradient = new RegularizedOrdinalGradient(hypothesis, learningRate);
    } else if (numClasses == 0 || numClasses == 1){
      hypothesis = new OLSHypothesis();
      gradient = new RegularizedDefaultGradient(hypothesis, learningRate);
    } else if (numClasses == 2) {
      hypothesis = new LogisticHypothesis();
      gradient = new RegularizedDefaultGradient(hypothesis, learningRate);
    } else {
      hypothesis = new SoftmaxHypothesis();
      gradient = new RegularizedSoftmaxGradient(hypothesis, learningRate);
    }

    if (sideInfo) {
      onUserSideLambda = hasOption(USER_SIDE_LAMBDA) ? Double.parseDouble(getOption(USER_SIDE_LAMBDA)) : 0;
      onItemSideLambda = hasOption(ITEM_SIDE_LAMBDA) ? Double.parseDouble(getOption(ITEM_SIDE_LAMBDA)) : 0;
      onDynamicSideLambda = hasOption(DYNAMIC_SIDE_LAMBDA) ? Double.parseDouble(getOption(DYNAMIC_SIDE_LAMBDA)) : 0;
      if (hasOption(USER_SIDE_INFO_FILE)) {
        userSideInfoFile = getOption(USER_SIDE_INFO_FILE);
        loadUserSideInfo(userSideInfoFile, featureVectorModel);
      }
      if (hasOption(ITEM_SIDE_INFO_FILE)) {
        itemSideInfoFile = getOption(ITEM_SIDE_INFO_FILE);
        loadItemSideInfo(itemSideInfoFile, featureVectorModel);
      }
      learner = new SideInfoAwareRecommenderLearner(gradient, hypothesis, featureVectorModel, biasLambda, factorsLambda, onUserSideLambda, onItemSideLambda, onDynamicSideLambda, cutsLambda);
    } else {
      learner = new JustRatingBasedRecommenderLearner(gradient, hypothesis, featureVectorModel, biasLambda, factorsLambda, cutsLambda);
    }

    dataModel = new FactorizationAwareDataModel(new GenericIncrementalDataModel(), learner);
    recommender = numClasses==1 ? new OnlineFactorizationRecommender(learner, dataModel, new ScoreOnTargetClassStrategy()) : new OnlineFactorizationRecommender(learner, dataModel, new MostProbableClassPredictionStrategy());
    Class<? extends Eval> evalClass = Class.forName(getOption(EVAL)).asSubclass(Eval.class);
    eval = evalClass.getConstructor(new Class[]{Recommender.class, int.class}).newInstance(recommender, numClasses);
    new GenericExperiment(new RecommenderEvalSetup(learner), eval, recommender, learner, numIterations, trainingFilePath, testFilePath, separator, ratingMapper, seeConvergence).run();
    return 0;
  }

  private void loadUserSideInfo(String userSideInfoFile, FeatureVectorModel featureVectorModel) throws IOException {
    logger.info("Loading user side information from "+userSideInfoFile);
    Path path = new Path(userSideInfoFile);
    Configuration conf = new Configuration();
    SequenceFileIterator<LongWritable, VectorWritable> iterator = new SequenceFileIterator<LongWritable, VectorWritable>(path, false, conf);
    while(iterator.hasNext()){
      Pair<LongWritable, VectorWritable> pair = iterator.next();
      featureVectorModel.setXs(pair.getFirst().get(), pair.getSecond().get());
    }
  }
  private void loadItemSideInfo(String itemSideInfoFile, FeatureVectorModel featureVectorModel) throws IOException {
    logger.info("Loading item side information from "+itemSideInfoFile);
    Path path = new Path(itemSideInfoFile);
    Configuration conf = new Configuration();
    SequenceFileIterator<LongWritable, VectorWritable> iterator = new SequenceFileIterator<LongWritable, VectorWritable>(path, false, conf);
    while(iterator.hasNext()){
      Pair<LongWritable, VectorWritable> pair = iterator.next();
      featureVectorModel.setTs(pair.getFirst().get(), pair.getSecond().get());
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new ExperimentDriver(), args);
  }
}
