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

public abstract class NumericalUserBasedEval extends NumericalEval{
  protected NumericalUserBasedEval(Recommender recommender) {
    super(recommender);
  }

  @Override
  public float[] aggregateScore(Map<Long, Map<Long, Float>> actualRatings, Map<Long, Map<Long,Float>> predictions ) throws TasteException {
    float totalScore = 0f;

    Set<Long> users = actualRatings.keySet();
    for(Long user:users){
      int size = actualRatings.get(user).size();
      totalScore += singleUserScore(user, actualRatings.get(user), predictions.get(user))[0]*size;
    }
    return new float[]{totalScore/users.size()};

  }
}
