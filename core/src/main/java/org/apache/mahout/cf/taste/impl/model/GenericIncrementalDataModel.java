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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * An in-memory, initially empty {@link org.apache.mahout.cf.taste.model.DataModel} which can be incrementally updated.
 */
public class GenericIncrementalDataModel extends AbstractDataModel {
  private static final Logger log = LoggerFactory.getLogger(GenericDataModel.class);

  private FastIDSet userIDs;
  private FastByIDMap<PreferenceArray> preferenceFromUsers;
  private FastIDSet itemIDs;
  private FastByIDMap<PreferenceArray> preferenceForItems;
  private float maxPreferenceValue = Float.NEGATIVE_INFINITY;
  private float minPreferenceValue = Float.POSITIVE_INFINITY;


  public GenericIncrementalDataModel() {

    userIDs = new FastIDSet();
    itemIDs = new FastIDSet();

    preferenceFromUsers = new FastByIDMap<PreferenceArray>();
    preferenceForItems = new FastByIDMap<PreferenceArray>();
  }

  @Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
    return userIDs.iterator();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long userID) throws TasteException {
    return preferenceFromUsers.get(userID);
  }

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    PreferenceArray preferenceArray = getPreferencesFromUser(userID);
    FastIDSet itemIDs = new FastIDSet();

    for(Preference preference:preferenceArray){
      itemIDs.add(preference.getItemID());
    }
    return itemIDs;
  }

  @Override
  public LongPrimitiveIterator getItemIDs() throws TasteException {
    return itemIDs.iterator();
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    return preferenceForItems.get(itemID);
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    PreferenceArray preferenceArray = getPreferencesFromUser(userID);

    for(Preference preference:preferenceArray){
      if (preference.getItemID()==itemID){
        return preference.getValue();
      }
    }
    return null;
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    throw new UnsupportedOperationException();
  }

  @Override
  public int getNumItems() throws TasteException {
    return itemIDs.size();
  }

  @Override
  public int getNumUsers() throws TasteException {
    return userIDs.size();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    return getPreferencesForItem(itemID).length();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    PreferenceArray preferenceArray1 = getPreferencesForItem(itemID1);
    PreferenceArray preferenceArray2 = getPreferencesForItem(itemID2);

    if(preferenceArray1==null || preferenceArray2==null){
      return 0;
    }
    FastIDSet users1 = new FastIDSet();
    FastIDSet users2 = new FastIDSet();

    for(Preference p: preferenceArray1){
      users1.add(p.getUserID());
    }
    for(Preference p: preferenceArray2){
      users2.add(p.getUserID());
    }
    return users1.intersectionSize(users2);

  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    userIDs.add(userID);
    itemIDs.add(itemID);
    if(value > maxPreferenceValue) {
      maxPreferenceValue = value;
      this.setMaxPreference(maxPreferenceValue);
    }

    if(value < minPreferenceValue) {
      minPreferenceValue = value;
      this.setMinPreference(minPreferenceValue);
    }

    if(preferenceFromUsers.containsKey(userID)&&preferenceFromUsers.get(userID).hasPrefWithItemID(itemID)){
      return;
    }

    PreferenceArray preferenceArray = null;
    if(preferenceFromUsers.containsKey(userID)){
      preferenceArray = preferenceFromUsers.get(userID);
    }
    else{
      preferenceArray = new GenericInitiallyEmptyUserPreferenceArray(userID);
    }
    preferenceArray.set(preferenceArray.length(),new GenericPreference(userID, itemID, value));
    preferenceFromUsers.put(userID, preferenceArray);


    PreferenceArray preferenceArrayForItem = null;
    if(preferenceForItems.containsKey(itemID)){
      preferenceArrayForItem = preferenceForItems.get(itemID);
    }
    else{
      preferenceArrayForItem = new GenericInitiallyEmptyItemPreferenceArray(itemID);
    }
    preferenceArrayForItem.set(preferenceArrayForItem.length(), new GenericPreference(userID, itemID, value));
    preferenceForItems.put(itemID, preferenceArrayForItem);
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean hasPreferenceValues() {
    return false;  //To change body of implemented methods use File | Settings | File Templates.
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {

  }
}
