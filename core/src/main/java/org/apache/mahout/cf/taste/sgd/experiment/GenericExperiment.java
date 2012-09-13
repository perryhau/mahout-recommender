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

package org.apache.mahout.cf.taste.sgd.experiment;

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import org.apache.commons.lang.ArrayUtils;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.sgd.common.RatingMapper;
import org.apache.mahout.cf.taste.sgd.eval.Eval;
import org.apache.mahout.cf.taste.sgd.eval.RecommenderEvalSetup;
import org.apache.mahout.cf.taste.sgd.learner.OnlineRecommenderLearner;
import org.apache.mahout.cf.taste.sgd.recommender.OnlineFactorizationRecommender;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * {@link OnlineFactorizationRecommender} experiment for validation and testing purposes
 */
public class GenericExperiment {
  private static Logger logger = LoggerFactory.getLogger(GenericExperiment.class);

  private Map<Long, Map<Long, Float>> actualRatings;
  private Map<Long, Map<Long, Float>> predictions;

  private RecommenderEvalSetup recommenderSetup;
  private Eval eval;
  private OnlineFactorizationRecommender recommender;
  private OnlineRecommenderLearner learner;
  private int numberOfIterations;
  private String trainRatingsFile;
  private String testRatingsFile;
  private String separator;
  private RatingMapper ratingMapper;
  private boolean seeConvergence;

  /**
   * @param recommenderSetup utility for easy setup
   * @param eval The specific {@link Eval}
   * @param recommender Underlying {@link OnlineFactorizationRecommender}
   * @param learner Underlying {@link OnlineRecommenderLearner}
   * @param numberOfIterations Number of iterations
   * @param trainRatingsFile Path for training file
   * @param testRatingsFile Path for test/validation file
   * @param separator separating String of userId,itemId,rating triplets
   * @param ratingMapper To convert incoming rating to a legitimate one
   * @param seeConvergence To see current score in every iteration. Useful for validation while selecting learning rate by seeing the status of convergence
   */
  public GenericExperiment(RecommenderEvalSetup recommenderSetup, Eval eval, OnlineFactorizationRecommender recommender, OnlineRecommenderLearner learner, int numberOfIterations, String trainRatingsFile, String testRatingsFile, String separator, RatingMapper ratingMapper, boolean seeConvergence) {
    this.actualRatings = new HashMap<Long, Map<Long, Float>>();
    this.predictions = new HashMap<Long, Map<Long, Float>>();

    this.recommenderSetup = recommenderSetup;
    this.eval = eval;
    this.recommender = recommender;
    this.learner = learner;
    this.numberOfIterations = numberOfIterations;
    this.trainRatingsFile = trainRatingsFile;
    this.testRatingsFile = testRatingsFile;
    this.separator = separator;
    this.ratingMapper = ratingMapper;
    this.seeConvergence = seeConvergence;
  }

  /**
   * Trains the recommender
   * @throws IOException
   * @throws TasteException
   */
  public void train()throws IOException, TasteException {
    for(int i = 0; i<numberOfIterations; i++){
      File trainingFile = new File(trainRatingsFile);
      for(String line: Files.readLines(trainingFile, Charsets.UTF_8)){
        String[] arr = line.split(separator);
        long userId = Long.parseLong(arr[0]);
        long itemId = Long.parseLong(arr[1]);
        float rating = ratingMapper.getRating(arr[2]);

        onlineTrain(userId, itemId, rating, null);
      }
      int iter = i+1;
      logger.info("Iteration "+iter);
      if (seeConvergence) {
        prepareTestData();
        System.out.println("Current score: "+ArrayUtils.toString(testAll()));
      }
    }
  }

  /**
   * Loads actual and predicted ratings of test data
   * @throws IOException
   * @throws TasteException
   */
  public void prepareTestData() throws IOException, TasteException {
    File testFile = new File(testRatingsFile);
    for(String line: Files.readLines(testFile, Charsets.UTF_8)){
      String[] arr = line.split(separator);
      long user = Long.parseLong(arr[0]);
      long item = Long.parseLong(arr[1]);
      float actualRating = ratingMapper.getRating(arr[2]);
      float predictedRating = recommender.estimatePreference(user, item);

      Map<Long, Float> ratingsOfUser;
      Map<Long, Float> predictedRatingsOfUser;
      if(actualRatings.containsKey(user)){
        ratingsOfUser = actualRatings.get(user);
        predictedRatingsOfUser = predictions.get(user);
      }

      else{
        ratingsOfUser = new HashMap<Long, Float>();
        predictedRatingsOfUser = new HashMap<Long, Float>();
      }

      ratingsOfUser.put(item, actualRating);
      predictedRatingsOfUser.put(item, predictedRating);
      actualRatings.put(user, ratingsOfUser);
      predictions.put(user, predictedRatingsOfUser);
    }
  }

  /**
   * Trains single example
   * @param user user id
   * @param item item id
   * @param rating rating
   * @param v if there is dynamic side information at the time that particular feedback provided
   * @throws TasteException
   */
  public void onlineTrain(long user, long item, float rating, Vector v) throws TasteException {
    if (v != null)
      recommenderSetup.setZ(user, item, v);
    recommender.setPreference(user, item, rating);
  }

  /**
   * @return Calculated user based scores
   * @throws TasteException
   */
  public Map<Long, float[]> testByIndividualUsers() throws TasteException {
    Map<Long, float[]> scores = new HashMap<Long, float[]>();

    Set<Long> testUsers = actualRatings.keySet();
    for(long user:testUsers){
      float[] userScore = eval.singleUserScore(user, actualRatings.get(user), predictions.get(user));
      scores.put(user, userScore);
    }
    return scores;
  }

  /**
   * @return single score of underlying {@link Eval} for entire test data
   * @throws TasteException
   */
  public float[] testAll() throws TasteException {
    logger.info("Final values of cuts are: "+ArrayUtils.toString(learner.getFeatureVectorModel().getCuts()));
    return eval.aggregateScore(actualRatings, predictions);
  }

  /**
   * runs the experiment
   * @throws Exception
   */
  public void run() throws Exception{
    train();
    prepareTestData();
    System.out.println("Score for each class\n"+ ArrayUtils.toString(testAll()));
  }
}
