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

package org.apache.mahout.cf.taste.impl.model;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.sgd.learner.OnlineRecommenderLearner;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.Vector;

import java.util.Collection;
import java.util.Map;

/**
 * A {@link DataModel} that trains underlying {@link OnlineRecommenderLearner} with a new preference, and delegates other functionality to the base model
 */
public class FactorizationAwareDataModel extends AbstractDataModel {
  private DataModel baseModel;
  private OnlineRecommenderLearner learner;

  public FactorizationAwareDataModel(DataModel baseModel, OnlineRecommenderLearner learner,
                                     Map<Long, Vector> userSideInfo, Map<Long, Vector> itemSideInfo,
                                     Map<Long, Map<Long, Vector>> jointFeaturesSoFar) throws TasteException {
    this.baseModel = baseModel;
    this.learner = learner;
    LongPrimitiveIterator iterator = baseModel.getUserIDs();
    while(iterator.hasNext()){
      long user = iterator.next();
      learner.initializeUserIfNeeded(user);
      if(userSideInfo != null && userSideInfo.containsKey(user)){
        learner.getFeatureVectorModel().setXs(user, userSideInfo.get(user));
      }
    }

    iterator = baseModel.getItemIDs();
    while(iterator.hasNext()){
      long item = iterator.next();
      learner.initializeItemIfNeeded(item);
      if(itemSideInfo != null && itemSideInfo.containsKey(item)){
        learner.getFeatureVectorModel().setTs(item, itemSideInfo.get(item));
      }
    }

    build(jointFeaturesSoFar);

  }

  public FactorizationAwareDataModel(DataModel baseModel, OnlineRecommenderLearner learner){
    this.baseModel = baseModel;
    this.learner = learner;
  }

  private void build(Map<Long, Map<Long, Vector>> jointFeaturesSoFar) throws TasteException {
    LongPrimitiveIterator iterator = baseModel.getUserIDs();
    while(iterator.hasNext()){
      long user = iterator.next();
      PreferenceArray preferences = baseModel.getPreferencesFromUser(user);
      for(Preference pref: preferences){
        long item = pref.getItemID();
        if(jointFeaturesSoFar != null){
          if(jointFeaturesSoFar.containsKey(user)){
            Map<Long, Vector> zContainingUser = jointFeaturesSoFar.get(user);
            if(zContainingUser.containsKey(item)){
              learner.getFeatureVectorModel().setZs(user, item, zContainingUser.get(item));
            }
          }
        }
        train(user, item, pref.getValue());
      }
    }

  }


  @Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
    return baseModel.getUserIDs();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws TasteException {
    return baseModel.getPreferencesFromUser(userID);
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    return baseModel.getItemIDsFromUser(userID);
  }

  @Override
  public LongPrimitiveIterator getItemIDs() throws TasteException {
    return baseModel.getItemIDs();
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    return baseModel.getPreferencesForItem(itemID);
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    return baseModel.getPreferenceValue(userID, itemID);
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    return baseModel.getPreferenceTime(userID, itemID);
  }

  @Override
  public int getNumItems() throws TasteException {
    return baseModel.getNumItems();
  }

  @Override
  public int getNumUsers() throws TasteException {
    return baseModel.getNumUsers();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    return baseModel.getNumUsersWithPreferenceFor(itemID);
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    return baseModel.getNumUsersWithPreferenceFor(itemID1, itemID2);
  }

  /**
   * trains the {@link OnlineRecommenderLearner}, and delegates persisting preference to underlying {@link org.apache.mahout.cf.taste.model.DataModel}.
   * It does not throw {@link org.apache.mahout.cf.taste.common.NoSuchItemException} or {@link org.apache.mahout.cf.taste.common.NoSuchUserException} anyway, but your data may not be persisted to the {@link org.apache.mahout.cf.taste.model.DataModel}.
   *
   * Additionally, before calling this method, you should:
   * 1- check if this is a new user and, if available, set x vector for her manually
   * 2- check if this is a new item and, if available, set t vector for it manually
   * 3- set z vector for this user-item pair manually
   *
   * @param userID user id
   * @param itemID item id
   * @param value preference value
   */
  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    try{
      baseModel.setPreference(userID, itemID, value);
    }
    catch(NoSuchItemException ignored){
    }
    catch(NoSuchUserException ignored){
    }
    catch(UnsupportedOperationException ignored){
    }

    finally{
      train(userID, itemID, value);
    }
  }

  private void train(long userID, long itemID, float value) {
    learner.train(userID, itemID, value);
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    baseModel.removePreference(userID, itemID);
  }

  @Override
  public boolean hasPreferenceValues() {
    return baseModel.hasPreferenceValues();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    baseModel.refresh(alreadyRefreshed);
  }
}
