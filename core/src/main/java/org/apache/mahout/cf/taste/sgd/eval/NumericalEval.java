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

public abstract class NumericalEval implements Eval{
  protected Recommender recommender;

  protected NumericalEval(Recommender recommender) {
    this.recommender = recommender;
  }

  public Recommender getRecommender() {
    return recommender;
  }

  public abstract float[] singleUserScore(long user, Map<Long, Float> actualRatings, Map<Long, Float> predictions) throws TasteException;


  public abstract float[] aggregateScore(Map<Long, Map<Long, Float>> actualRatings, Map<Long, Map<Long,Float>> predictions) throws TasteException;


  public float ratingError(long user, long item, float y) throws TasteException {
    return Math.abs(recommender.estimatePreference(user, item) - y);
  }
  public float ratingError(float actual, float predicted){
    return Math.abs(actual-predicted);
  }

}
