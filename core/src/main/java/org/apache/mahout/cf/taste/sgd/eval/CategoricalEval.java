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

package org.apache.mahout.cf.taste.sgd.eval;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.util.Map;
import java.util.Set;

/**
 * In this evaluation setting, category predictions of the test user-item pairs, and actual test ratings, each of which is actually rated by the user,
 * should be provided.
 * Then standard computation is done.
 * (global item set is considered as the items in the test set, meaning different set for different users)
 * The measures are computed per user basis, the average is considered as the final score.
 */
public abstract class CategoricalEval implements Eval{
  protected Recommender recommender;
  protected int numberOfCategories;

  protected CategoricalEval(Recommender recommender, int numberOfCategories) {
    this.recommender = recommender;
    this.numberOfCategories = numberOfCategories;
  }

  protected float[][] precisionAndRecall(long user, Map<Long, Float> testRatings, Map<Long, Float> predictedRatings)throws TasteException {
    float[][] precisionAndRecall = new float[2][numberOfCategories];
    int[] tp = new int[numberOfCategories];
    int[] relevant = new int[numberOfCategories];
    int[] recommended = new int[numberOfCategories];
    float[] precision = new float[numberOfCategories];
    float[] recall = new float[numberOfCategories];

    for(int i = 0; i<numberOfCategories; i++){
      tp[i] = 0;
      relevant[i] = 0;
      recommended[i] = 0;
      precision[i] = 0.0f;
      recall[i] = 0.0f;
    }

    for(Map.Entry<Long, Float> testRating: testRatings.entrySet()){
      float actual =  testRating.getValue();
      float prediction = predictedRatings.get(testRating.getKey());

      relevant[(int)actual]+=1;
      recommended[(int)prediction]+=1;
      if(actual==prediction){
        tp[(int)prediction]++;
      }
    }

    for(int i = 0; i<numberOfCategories; i++){
      precision[i] = recommended[i] > 0 ? (float)tp[i]/recommended[i] : 0;
      recall[i] = relevant[i] > 0 ? (float)tp[i]/relevant[i] : 0;
    }
    precisionAndRecall[0] = precision;
    precisionAndRecall[1] = recall;

    return precisionAndRecall;
  }



  public abstract float[] singleUserScore(long user, Map<Long, Float> testRatings, Map<Long, Float> predictedRatings) throws TasteException;


  public float[] aggregateScore(Map<Long, Map<Long, Float>> testRatings, Map<Long, Map<Long, Float>> predictedRatings) throws TasteException {
    float[] score = new float[numberOfCategories];

    Set<Long> users = testRatings.keySet();
    for(long user:users){
      float[] scoreOfUser = singleUserScore(user, testRatings.get(user), predictedRatings.get(user));
      for(int i = 0; i<numberOfCategories; i++){
        score[i]+=scoreOfUser[i];
      }
    }
    for(int i = 0; i<numberOfCategories; i++){
      score[i] = score[i]/users.size();
    }
    return score;
  }

}
