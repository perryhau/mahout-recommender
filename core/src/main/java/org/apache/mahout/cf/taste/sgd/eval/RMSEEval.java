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

import org.apache.mahout.cf.taste.recommender.Recommender;

public class RMSEEval extends NumericalRatingBasedEval{

  public RMSEEval(Recommender recommender) {
    super(recommender);
  }

  @Override
  protected float[] aggregateFunc(float v) {
    return new float[]{(float)Math.sqrt(v)};
  }

  @Override
  protected float ratingFunc(float v) {
    return (float)Math.pow(v, 2);
  }

}
