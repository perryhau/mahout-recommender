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

import org.apache.mahout.cf.taste.sgd.learner.OnlineRecommenderLearner;
import org.apache.mahout.cf.taste.sgd.model.FeatureVectorModel;
import org.apache.mahout.math.Vector;

import java.util.Map;

/**
 * Utility functions for interacting with {@link FeatureVectorModel}
 */
public class RecommenderEvalSetup {
  FeatureVectorModel featureModel;

  public RecommenderEvalSetup(OnlineRecommenderLearner learner) {
    featureModel = learner.getFeatureVectorModel();
  }

  public void setX(long user, Vector x){
    if(!featureModel.xSetAlready(user))
      featureModel.setXs(user, x);
  }

  public void setT(long item, Vector t){
    if(!featureModel.tSetAlready(item))
      featureModel.setTs(item, t);
  }

  public void setZ(long user, long item, Vector z){
    if(!featureModel.zSetAlready(user, item))
      featureModel.setZs(user, item, z);
  }

  public void setXs(Map<Long, Vector> xs){
    for(long user:xs.keySet()){
      setX(user, xs.get(user));
    }
  }

  public void setTs(Map<Long, Vector> ts){
    for(long item:ts.keySet()){
      setT(item, ts.get(item));
    }
  }

  public void setZs(Map<Long, Map<Long, Vector>> zs){
    for(long user: zs.keySet()){
      Map<Long, Vector> withItems = zs.get(user);
      setOneUserZ(user, withItems);
    }
  }

  public void setOneUserZ(long user,Map<Long, Vector> zs){
    for(long item:zs.keySet()){
      setZ(user, item, zs.get(item));
    }

  }

}
