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

package org.apache.mahout.cf.taste.sgd.model;


import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Random;

/**
 * Base class for data store for alphas, betas, side info, side info parameters and ordinal cuts.
 */
public abstract class FeatureVectorModel {
  protected int classes;
  protected int classSizeForVectors;
  protected int factorSize;
  protected boolean ordinal = false;
  protected double[] cuts;


  /**
   * @return number of classes if there is more than 1 classes
   */
  public int numberOfClasses(){
    return (classes == 0 || classes == 1)?1:classes;
  }

  /**
   * initializes alphas for the specified user randomly
   * @param userId user id to initialize alphas for
   */
  public void initializeUserIfNeeded(long userId){
    if(!checkUser(userId)){
      Vector[] alphas = initializeRandom(factorSize+3, classSizeForVectors,1);
      setAlphas(userId, alphas);
    }
  }

  /**
   * initializes betas for the specified item randomly
   * @param itemId item id to initialize betas for
   */
  public void initializeItemIfNeeded(long itemId){
    if(!checkItem(itemId)){
      Vector[] betas = initializeRandom(factorSize+3, classSizeForVectors,2);
      setBetas(itemId, betas);
    }
  }

  /**
   * random vector initializer
   * @param interceptIndex value on this index is set to 1
   */
  protected Vector[] initializeRandom(int size, int classes, int interceptIndex){
    Vector[] features;
    double[][] values;
    int c;
    if(classes == 0 || classes == 1 || classes == 2){
      c = 1;
    }
    else{
      c = classes;
    }
    features = new Vector[c];
    values = new double[c][];

    for(int i = 0; i<c; i++){
      values[i] = new double[size];
      values[i][interceptIndex] = 1;
      values[i][0]=0;
      Random r = RandomUtils.getRandom();
      for(int j = 1; j<size; j++){
        if(j!=interceptIndex){
          values[i][j] = r.nextDouble()/10 +0.1;
        }
      }
      features[i] = new DenseVector(values[i]);
    }
    return features;
  }

  /**
   * all zeros vector initializer
   */
  protected Vector[] initializeZeros(int size, int classes){
    Vector[] features;
    double[][] values;
    int c;
    if (classes == 0 || classes == 1 || classes == 2){
      c = 1;
    }
    else c = classes;

    features = new Vector[c];
    values = new double[c][];

    for(int i = 0; i<c; i++){
      values[i] =  new double[size];
      for (int j = 0; j<size; c++){
        values[i][j] = 0;
      }
      features[i] = new DenseVector(values[i]);
    }
    return features;
  }

  /**
   * @return ordinal cuts
   */
  public abstract double[] getCuts();

  /**
   * @param userId the user id
   * @return alphas for specified user
   */
  public abstract Vector[] getAlphas(long userId);

  /**
   * @param itemId the item id
   * @return betas for specified item
   */
  public abstract Vector[] getBetas(long itemId);

  /**
   * @param userId the user id
   * @return parameters of user on item side info
   */
  public abstract Vector[] getThetasOnTs(long userId);

  /**
   * @param userId the user id
   * @return parameters of user on dynamic side info
   */
  public abstract Vector[] getThetasOnZs(long userId);

  /**
   * @param itemId the item id
   * @return parameters of item on user side info
   */
  public abstract Vector[] getGammasOnXs(long itemId);

  /**
   * @param itemId the item id
   * @return parameters of item on dynamic side info
   */
  public abstract Vector[] getGammasOnZs(long itemId);

  /**
   * @param userId the user id
   * @return user side info vector
   */
  public abstract Vector getXs(long userId);

  /**
   * @param itemId the item id
   * @return item side info vector
   */
  public abstract Vector getTs(long itemId);

  /**
   * @param userId the user id
   * @param itemId the item id
   * @return dynamic side info vector at the feedback moment from user to item
   */
  public abstract Vector getZs(long userId, long itemId);

  /**
   * @param cuts ordinal cuts
   */
  public abstract void setCuts(double[] cuts);

  /**
   * @param index index to set cut
   * @param cut cut
   */
  public abstract void setCut(int index, double cut);

  /**
   * @param userId the user id
   * @param alphas factors for the user
   */
  public abstract void setAlphas(long userId, Vector[] alphas);

  /**
   * @param itemId the item id
   * @param betas factors for the item
   */
  public abstract void setBetas(long itemId, Vector[] betas);

  /**
   * @param userId the user id
   * @param thetasOnTs parameters of the user on item side info
   */
  public abstract void setThetasOnTs(long userId, Vector[] thetasOnTs);

  /**
   * @param userId the user id
   * @param thetasOnZs parameters of the user on dynamic side info
   */
  public abstract void setThetasOnZs(long userId, Vector[] thetasOnZs);

  /**
   * @param itemId the item id
   * @param gammasOnXs parameters of the item on user side info
   */
  public abstract void setGammasOnXs(long itemId, Vector[] gammasOnXs);

  /**
   * @param itemId the item id
   * @param gammasOnZs parameters of the item on dynamic side info
   */
  public abstract void setGammasOnZs(long itemId, Vector[] gammasOnZs);

  /**
   * @param userId the user id
   * @param x user side info
   */
  public abstract void setXs(long userId, Vector x);

  /**
   * @param itemId the item id
   * @param t item side info
   */
  public abstract void setTs(long itemId, Vector t);

  /**
   * note that this does not have to be stored. If you do with a bulk import, please keep it temporary. See {@link #removeZs(long, long)}
   * @param userId the user id
   * @param itemId the item id
   * @param z dynamic side info in the feedback moment
   */
  public abstract void setZs(long userId, long itemId, Vector z);

  /**
   * @param userId the user id
   * @return true if this user is initialized properly
   */
  public abstract boolean checkUser(long userId);

  /**
   * @param itemId the item id
   * @return true if this item is initialized properly
   */
  public abstract boolean checkItem(long itemId);

  /**
   * @param userId the user id
   * @return true if user side info is set already
   */
  public abstract boolean xSetAlready(long userId);

  /**
   * @param itemId the item id
   * @return true if item side info is set already
   */
  public abstract boolean tSetAlready(long itemId);

  /**
   * @param userId the user id
   * @param itemId the item id
   * @return if dynamic side info for this user-item pair is set already
   */
  public abstract boolean zSetAlready(long userId, long itemId);

  /**
   * removes dynamic side info for the feedback moment of user on item
   * @param user the user id
   * @param item the item id
   */
  public abstract void removeZs(long user, long item);
}
