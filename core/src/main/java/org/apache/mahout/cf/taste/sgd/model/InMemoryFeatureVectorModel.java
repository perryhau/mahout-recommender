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
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * In-memory {@link FeatureVectorModel}
 */
public class InMemoryFeatureVectorModel  extends FeatureVectorModel{

  private Map<Long, Vector[]> alphas;
  private Map<Long, Vector[]> betas;
  private Map<Long, Vector[]> thetasOnTs;
  private Map<Long, Vector[]> thetasOnZs;
  private Map<Long, Vector[]> gammasOnXs;
  private Map<Long, Vector[]> gammasOnZs;
  private Map<Long, Vector> xs;
  private Map<Long, Vector> ts;
  private Map<Long, Map<Long, Vector>> zs;

  /**
   * Constructor
   * @param factorSize number of factors
   * @param classes number of classes
   * @param ordinal true if this is for an ordinal recommender
   */
  public InMemoryFeatureVectorModel(int factorSize, int classes, boolean ordinal) {
    alphas = new HashMap<Long, Vector[]>();
    betas = new HashMap<Long, Vector[]>();
    thetasOnTs = new HashMap<Long, Vector[]>();
    thetasOnZs = new HashMap<Long, Vector[]>();
    gammasOnXs = new HashMap<Long, Vector[]>();
    gammasOnZs = new HashMap<Long, Vector[]>();
    this.ordinal = ordinal;
    xs = new HashMap<Long, Vector>();
    ts = new HashMap<Long, Vector>();
    zs = new HashMap<Long, Map<Long, Vector>>();
    this.factorSize = factorSize;
    this.classes = classes;
    if(this.ordinal){
      this.classSizeForVectors = 1;
      cuts = new double[classes-1];
      Random r = RandomUtils.getRandom();
      double first = r.nextDouble()/10+0.1;
      cuts[0] = first;
      double step = 0.9/cuts.length;
      for(int i = 1; i<cuts.length; i++){
        cuts[i] = step;
      }
    }
    else{
      this.classSizeForVectors = classes<=2?1:classes;
      cuts = new double[classes-1];
      Arrays.fill(cuts, 0);
    }
  }

  /**
   * ordinal is set to false by default
   * @param factorSize number of factors
   * @param classes number of classes
   */
  public InMemoryFeatureVectorModel(int factorSize, int classes){
    this(factorSize, classes, false);
  }


  /**
   * ordinal is set to false, and classes is set to 1 by default
   * @param factorSize number of factors
   */
  public InMemoryFeatureVectorModel(int factorSize){
    this(factorSize, 1, false);
  }

  public InMemoryFeatureVectorModel(Map<Long, Vector[]> alphas, Map<Long, Vector[]> betas, Map<Long, Vector[]> thetasOnTs, Map<Long, Vector[]> thetasOnZs, Map<Long, Vector[]> gammasOnXs, Map<Long, Vector[]> gammasOnZs, Map<Long, Vector> xs, Map<Long, Vector> ts, Map<Long, Map<Long, Vector>> zs, int classes) {

    this.alphas = alphas;
    this.betas = betas;
    this.thetasOnTs = thetasOnTs;
    this.thetasOnZs = thetasOnZs;
    this.gammasOnXs = gammasOnXs;
    this.gammasOnZs = gammasOnZs;
    this.xs = xs;
    this.ts = ts;
    this.zs = zs;
    this.cuts = new double[classes-1];
    this.classes = classes;
    this.classSizeForVectors = classes<=2?1:classes;
    factorSize = alphas.entrySet().iterator().next().getValue()[0].size();
  }

  public InMemoryFeatureVectorModel(Map<Long, Vector[]> alphas, Map<Long, Vector[]> betas, Map<Long, Vector[]> thetasOnTs, Map<Long, Vector[]> thetasOnZs, Map<Long, Vector[]> gammasOnXs, Map<Long, Vector[]> gammasOnZs, Map<Long, Vector> xs, Map<Long, Vector> ts, Map<Long, Map<Long, Vector>> zs, double[] cuts){
    this.alphas = alphas;
    this.betas = betas;
    this.thetasOnTs = thetasOnTs;
    this.thetasOnZs = thetasOnZs;
    this.gammasOnXs = gammasOnXs;
    this.gammasOnZs = gammasOnZs;
    this.xs = xs;
    this.ts = ts;
    this.zs = zs;
    this.cuts = cuts;
    classes = cuts.length+1;
    this.classSizeForVectors = 1;
    factorSize = alphas.entrySet().iterator().next().getValue()[0].size();
  }


  @Override
  public double[] getCuts() {
    return this.cuts;
  }

  @Override
  public Vector[] getAlphas(long userId) {
    initializeUserIfNeeded(userId);
    return alphas.get(userId);
  }
  @Override
  public Vector[] getBetas(long itemId) {
    initializeItemIfNeeded(itemId);
    return betas.get(itemId);
  }

  @Override
  public Vector[] getThetasOnTs(long userId) {
    return thetasOnTs.containsKey(userId)?thetasOnTs.get(userId):initializeVectorArray();
  }

  @Override
  public Vector[] getThetasOnZs(long userId) {
    return thetasOnZs.containsKey(userId)?thetasOnZs.get(userId):initializeVectorArray();
  }

  @Override
  public Vector[] getGammasOnXs(long itemId) {
    return gammasOnXs.containsKey(itemId)?gammasOnXs.get(itemId):initializeVectorArray();
  }

  @Override
  public Vector[] getGammasOnZs(long itemId) {
    return gammasOnZs.containsKey(itemId)?gammasOnZs.get(itemId):initializeVectorArray();
  }


  private Vector[] initializeVectorArray(){
    Vector[] initial = new Vector[classSizeForVectors<=2?1:classSizeForVectors];
    for(int i = 0; i<initial.length; i++){
      initial[i] = new RandomAccessSparseVector(0);
    }
    return initial;
  }

  @Override
  public Vector getXs(long userId) {
    return xs.containsKey(userId)?xs.get(userId):new RandomAccessSparseVector(0);
  }

  @Override
  public Vector getTs(long itemId) {
    return ts.containsKey(itemId)?ts.get(itemId):new RandomAccessSparseVector(0);
  }

  @Override
  public Vector getZs(long userId, long itemId) {
    if (zs.containsKey(userId)){
      return zs.get(userId).containsKey(itemId)?zs.get(userId).get(itemId):new RandomAccessSparseVector(0);
    }
    return new RandomAccessSparseVector(0);
  }

  @Override
  public void setCuts(double[] cuts) {
    this.cuts = cuts;
  }

  @Override
  public void setCut(int index, double cut) {
    this.cuts[index] = cut;
  }

  @Override
  public void setAlphas(long userId, Vector[] alphas) {
    this.alphas.put(userId, alphas);
  }

  @Override
  public void setBetas(long itemId, Vector[] betas) {
    this.betas.put(itemId, betas);
  }

  @Override
  public void setThetasOnTs(long userId, Vector[] thetasOnTs) {
    this.thetasOnTs.put(userId, thetasOnTs);
  }

  @Override
  public void setThetasOnZs(long userId, Vector[] thetasOnZs) {
    this.thetasOnZs.put(userId, thetasOnZs);
  }

  @Override
  public void setGammasOnXs(long itemId, Vector[] gammasOnXs) {
    this.gammasOnXs.put(itemId, gammasOnXs);
  }

  @Override
  public void setGammasOnZs(long itemId, Vector[] gammasOnZs) {
    this.gammasOnZs.put(itemId, gammasOnZs);
  }


  @Override
  public void setXs(long userId, Vector x) {
    this.xs.put(userId, x);
  }

  @Override
  public void setTs(long itemId, Vector t) {
    this.ts.put(itemId, t);
  }

  @Override
  public void setZs(long userId, long itemId, Vector z) {
    if(zs.containsKey(userId)){
      zs.get(userId).put(itemId, z);
    }

    else{
      Map<Long, Vector> zMap = new HashMap<Long, Vector>();
      zMap.put(itemId, z);
      this.zs.put(userId, zMap);
    }
  }

  @Override
  public boolean checkUser(long userId) {
    return alphas.containsKey(userId);
  }

  @Override
  public boolean checkItem(long itemId) {
    return betas.containsKey(itemId);
  }

  @Override
  public boolean xSetAlready(long userId) {
    return xs.containsKey(userId);
  }

  @Override
  public boolean tSetAlready(long itemId) {
    return ts.containsKey(itemId);
  }

  @Override
  public boolean zSetAlready(long userId, long itemId) {
    return zs.containsKey(userId)&&zs.get(userId).containsKey(itemId);
  }

  @Override
  public void removeZs(long userId, long itemId) {
    if(zs.containsKey(userId)&&zs.get(userId).containsKey(itemId)){
      zs.get(userId).remove(itemId);
    }
  }
}
