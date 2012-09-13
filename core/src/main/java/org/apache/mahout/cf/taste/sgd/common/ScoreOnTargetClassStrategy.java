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

package org.apache.mahout.cf.taste.sgd.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ScoreOnTargetClassStrategy implements RatingPredictionStrategy{
  private static Logger logger = LoggerFactory.getLogger(ScoreOnTargetClassStrategy.class);

  public ScoreOnTargetClassStrategy() {
    logger.info("Will compute score as the probability on target class, or as the numerical prediction.");
  }

  /**
   * @param classDistribution prediction by {@link org.apache.mahout.cf.taste.sgd.learner.OnlineRecommenderLearner#predictFull(long, long)}
   * @param targetClass target category index
   * @return returns the score on target category. This is the prediction score in numerical recommender, or the probability on target category if this is categorical/ordinal
   */
  @Override
  public float predict(double[] classDistribution, int targetClass) {
    return (float)classDistribution[targetClass];
  }
}
