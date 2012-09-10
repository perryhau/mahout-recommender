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

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import com.google.common.primitives.Longs;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.iterator.CountingIterator;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * like {@link GenericItemPreferenceArray}, but this is initially empty, and {@link java.util.List} based.
 * Most of the methods are just copies of the original {@link GenericItemPreferenceArray} class.
 */
public class GenericInitiallyEmptyItemPreferenceArray implements PreferenceArray {
  private static final int USER = 0;
  private static final int VALUE = 2;
  private static final int VALUE_REVERSED = 3;

  private ArrayList<Long> ids;
  private long id;
  private ArrayList<Float> values;

  public GenericInitiallyEmptyItemPreferenceArray(long id) {
    this.ids = new ArrayList<Long>();
    values = new ArrayList<Float>();
    this.id = id;
  }



  @Override
  public int length() {
    return ids.size();
  }

  @Override
  public Preference get(int i) {
    return new PreferenceView(i);
  }

  @Override
  public void set(int i, Preference pref) {
    id = pref.getItemID();
    ids.add(pref.getUserID());
    values.add(pref.getValue());
  }

  @Override
  public long getUserID(int i) {
    return ids.get(i);
  }

  @Override
  public void setUserID(int i, long userID) {
    ids.set(i,userID);
  }

  @Override
  public long getItemID(int i) {
    return id;
  }

  /**
   * {@inheritDoc}
   *
   * Note that this method will actually set the item ID for <em>all</em> preferences.
   */
  @Override
  public void setItemID(int i, long itemID) {
    id = itemID;
  }

  /**
   * @return all user IDs
   */
  @Override
  public long[] getIDs() {
    return Longs.toArray(ids);
  }

  @Override
  public float getValue(int i) {
    return values.get(i);
  }

  @Override
  public void setValue(int i, float value) {
    values.set(i,value);
  }

  @Override
  public void sortByUser() {
    selectionSort(USER);
  }

  @Override
  public void sortByItem() { }

  @Override
  public void sortByValue() {
    selectionSort(VALUE);
  }

  @Override
  public void sortByValueReversed() {
    selectionSort(VALUE_REVERSED);
  }

  @Override
  public boolean hasPrefWithUserID(long userID) {
    for (long id : ids) {
      if (userID == id) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean hasPrefWithItemID(long itemID) {
    return id == itemID;
  }

  private void selectionSort(int type) {
    int max = length();
    boolean sorted = true;
    for (int i = 1; i < max; i++) {
      if (isLess(i, i - 1, type)) {
        sorted = false;
        break;
      }
    }
    if (sorted) {
      return;
    }
    for (int i = 0; i < max; i++) {
      int min = i;
      for (int j = i + 1; j < max; j++) {
        if (isLess(j, min, type)) {
          min = j;
        }
      }
      if (i != min) {
        swap(i, min);
      }
    }
  }

  private boolean isLess(int i, int j, int type) {
    switch (type) {
      case USER:
        return ids.get(i) < ids.get(j);
      case VALUE:
        return values.get(i) < values.get(j);
      case VALUE_REVERSED:
        return values.get(i) >= values.get(j);
      default:
        throw new IllegalStateException();
    }
  }

  private void swap(int i, int j) {
    long temp1 = ids.get(i);
    float temp2 = values.get(i);
    ids.set(i, ids.get(j));
    values.set(i, values.get(j));
    ids.set(j, temp1);
    values.set(j, temp2);
  }


  @Override
  public Iterator<Preference> iterator() {
    return Iterators.transform(new CountingIterator(length()),
        new Function<Integer, Preference>() {
          @Override
          public Preference apply(Integer from) {
            return new PreferenceView(from);
          }
        });
  }


  @Override
  public PreferenceArray clone() {
    throw new UnsupportedOperationException();
  }


  private final class PreferenceView implements Preference {

    private final int i;

    private PreferenceView(int i) {
      this.i = i;
    }

    @Override
    public long getUserID() {
      return GenericInitiallyEmptyItemPreferenceArray.this.getUserID(i);
    }

    @Override
    public long getItemID() {
      return GenericInitiallyEmptyItemPreferenceArray.this.getItemID(i);
    }

    @Override
    public float getValue() {
      return values.get(i);
    }

    @Override
    public void setValue(float value) {
      values.set(i, value);
    }

  }
}
