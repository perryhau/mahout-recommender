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

package org.apache.mahout.cf.taste.sgd.recommender;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.PreferredItemsNeighborhoodCandidateItemsStrategy;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.sgd.learner.OnlineRecommenderLearner;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.sgd.model.FeatureVectorModel;

import java.util.Collection;
import java.util.List;

/**
 * {@link org.apache.mahout.cf.taste.recommender.Recommender} wrapper on {@link OnlineRecommenderLearner}
 */
public class OnlineFactorizationRecommender extends AbstractRecommender {
  private OnlineRecommenderLearner recommenderLearner;
  /**
   * The target category index, do not set this if this is a numerical recommender.
   * By default, this is the last category (the category with the highest integer value).
   * One needs to define the target category to use the recommender, otherwise. Normally the {@link org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis}
   * estimates the probability distribution for all categories, however, this was the only way I could find to stick to Taste
   * interface.
   */
  private int classIndice;

  private FeatureVectorModel model;

  /**
   * @param recommenderLearner underlying {@link OnlineRecommenderLearner}
   * @param dataModel underlying {@link DataModel}
   */
  public OnlineFactorizationRecommender(OnlineRecommenderLearner recommenderLearner, DataModel dataModel){
    this(recommenderLearner, dataModel, new PreferredItemsNeighborhoodCandidateItemsStrategy());
  }

  /**
   * @param classIndex The target category index, do not set this if this is a numerical recommender.
   *                    By default, this is the last category (the category with the highest integer value).
   *                    One needs to define the target category to use the recommender, otherwise. Normally the {@link org.apache.mahout.cf.taste.sgd.hypothesis.Hypothesis}
   *                    estimates the probability distribution for all categories, however, this was the only way I could find to stick to Taste interface.
   * @param recommenderLearner underlying {@link OnlineRecommenderLearner}
   * @param dataModel underlying {@link DataModel}
   */
  public OnlineFactorizationRecommender(int classIndex, OnlineRecommenderLearner recommenderLearner, DataModel dataModel){
    this(classIndex, recommenderLearner, dataModel, new PreferredItemsNeighborhoodCandidateItemsStrategy());
  }

  protected OnlineFactorizationRecommender(OnlineRecommenderLearner recommenderLearner, DataModel dataModel, CandidateItemsStrategy candidateItemsStrategy) {
    this(recommenderLearner.getFeatureVectorModel().numberOfClasses()-1, recommenderLearner, dataModel, candidateItemsStrategy);
  }

  protected OnlineFactorizationRecommender(int classIndex, OnlineRecommenderLearner recommenderLearner, DataModel dataModel, CandidateItemsStrategy candidateItemsStrategy) {
    super(dataModel, candidateItemsStrategy);
    this.classIndice = classIndex;
    this.recommenderLearner = recommenderLearner;
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException {
    PreferenceArray preferencesOfUser = getDataModel().getPreferencesFromUser(userID);
    FastIDSet possibleItems = getAllOtherItems(userID, preferencesOfUser);

    return TopItems.getTopItems(howMany, possibleItems.iterator(), rescorer, new PreferenceEstimator(userID));
  }

  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    return (float)recommenderLearner.predictFull(userID, itemID)[classIndice];
  }

  /**
   * @param userID the user id
   * @param itemID the item id
   * @return whole class distribution of scores
   * @throws TasteException if something goes wrong
   */
  public double[] predictAll(long userID, long itemID) throws TasteException {
    return recommenderLearner.predictFull(userID, itemID);
  }


  @Override
  public void refresh(Collection<Refreshable> refreshables) {
  }

  private class PreferenceEstimator implements TopItems.Estimator<Long>{
    private long theUserID;

    private PreferenceEstimator(long theUserID) {
      this.theUserID = theUserID;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      return estimatePreference(theUserID, itemID);
    }
  }
}
