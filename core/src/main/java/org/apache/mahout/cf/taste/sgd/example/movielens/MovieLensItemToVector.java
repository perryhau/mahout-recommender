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

package org.apache.mahout.cf.taste.sgd.example.movielens;

import org.apache.hadoop.io.Text;
import org.apache.mahout.cf.taste.sgd.common.ToVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.Map;

/**
 * Simple vectorizer for movies in MovieLens-1M data
 */
public class MovieLensItemToVector implements ToVector{
  private Map<String, Integer> genreMap;

  public MovieLensItemToVector() {
    String[] genres = {
        "action",
        "adventure",
        "animation",
        "children's",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film-noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci-fi",
        "thriller",
        "war",
        "western"};

    genreMap = new HashMap<String, Integer>();
    for(int i = 0; i<genres.length; i++){
      genreMap.put(genres[i], i+2);
    }
  }

  @Override
  public Vector get(Text text) {
    String[] arr = text.toString().split("::");
    String year = arr[1].substring(arr[1].lastIndexOf("(") + 1, arr[1].lastIndexOf(")"));
    int thisYear = new GregorianCalendar().get(Calendar.YEAR);
    int timePassed = thisYear-Integer.parseInt(year);

    Vector v = new RandomAccessSparseVector(0);
    String[] movieGenres = arr[2].split("\\|");

    for(String genre:movieGenres){
      int genreId = genreMap.get(genre.toLowerCase());
      v.setQuick(genreId, 1);
    }
    v.setQuick(1, (double)timePassed/100);

    v.setQuick(0,1);
    return v;
  }
}
