package org.apache.mahout.cf.taste.sgd.example.movielens;
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

import org.apache.hadoop.io.Text;
import org.apache.mahout.cf.taste.sgd.common.ToVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.HashMap;
import java.util.Map;

/**
 * Simple vectorizer for users in MovieLens-1M data
 */
public class MovieLensUserToVector implements ToVector{
  private Map<String, Integer> genderMap;
  private Map<Integer, Integer> occupationMap;

  public MovieLensUserToVector(){

    this.genderMap = new HashMap<String, Integer>();
    this.occupationMap = new HashMap<Integer, Integer>();

    genderMap.put("F", 1);
    genderMap.put("M", 2);

    for(int i=0; i<=20; i++){
      occupationMap.put(i, i+4);
    }
  }

  @Override
  public Vector get(Text text) {
    String[] arr = text.toString().split("::");

    Vector v = new RandomAccessSparseVector(0);
    v.setQuick(genderMap.get(arr[1]), 1);
    v.setQuick(3, Double.parseDouble(arr[2])/100);
    v.setQuick(occupationMap.get(Integer.parseInt(arr[3])), 1);

    int zip = arr[4].contains("-")?Integer.parseInt(arr[4].substring(0, arr[4].indexOf("-"))):Integer.parseInt(arr[4]);
    v.setQuick(zip, 1);

    v.setQuick(0,1);

    return v;
  }
}
