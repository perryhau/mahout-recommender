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

public abstract class NumericalRatingBasedEval extends NumericalEval{
  protected NumericalRatingBasedEval(Recommender recommender) {
    super(recommender);
  }

  @Override
  public float[] aggregateScore(Map<Long, Map<Long, Float>> actualRatings, Map<Long, Map<Long,Float>> predictions ) throws TasteException {
    float totalScore = 0f;
    int totalSize = 0;

    Set<Long> users = actualRatings.keySet();
    for(Long user:users){
      Map<Long, Float> ratings = actualRatings.get(user);
      for(Map.Entry<Long, Float> rating:ratings.entrySet()){
        totalScore+= ratingFunc(ratingError(rating.getValue(), predictions.get(user).get(rating.getKey())));
      }
      int size = actualRatings.get(user).size();
      totalSize += size;
    }
    return aggregateFunc(totalScore / totalSize);
  }

  @Override
  public float[] singleUserScore(long user, Map<Long, Float> actualRatings, Map<Long, Float> predictions) throws TasteException {
    float totalError = 0f;
    for(Map.Entry<Long, Float> item:actualRatings.entrySet()){
      totalError += ratingFunc(ratingError(item.getValue(), predictions.get(item.getKey())));
    }
    return aggregateFunc(totalError/actualRatings.size());

  }

  protected abstract float[] aggregateFunc(float v);

  protected abstract float ratingFunc(float v);

}
