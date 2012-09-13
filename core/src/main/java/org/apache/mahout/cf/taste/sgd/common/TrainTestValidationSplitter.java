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

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

import java.io.File;
import java.io.PrintWriter;
import java.util.Random;

/**
 * Splits text file of lines of feedback into 3 files: training, validation, test
 * By default, 15% of data is for test, and 10% of data is for validation
 */
public class TrainTestValidationSplitter extends AbstractJob {
  public static String INPUT = "input";
  public static String RANDOM_SEED = "randomSeed";
  public static String TEST_PERCENT = "testPercent";
  public static String VALIDATION_PERCENT = "validationPercent";
  public static String TRAIN_OUT = "trainOut";
  public static String TEST_OUT = "testOut";
  public static String VALIDATION_OUT = "validationOut";

  @Override
  public int run(String[] args) throws Exception {
    addOption(INPUT, "i", "Input path", true);
    addOption(TEST_PERCENT, "tp", "Percentage of entire data to spare for test, default 0.15", false);
    addOption(VALIDATION_PERCENT, "vp", "Percentage of entire data to spare for test, default 0.10", false);
    addOption(TRAIN_OUT, "train", "Path to training split", true);
    addOption(VALIDATION_OUT, "validation", "Path to validation split", true);
    addOption(TEST_OUT, "test", "Path to test split", true);
    addOption(RANDOM_SEED, "seed", "Random seed", false);
    addOption(DefaultOptionCreator.helpOption());

    parseArguments(args);

    File input = new File(getOption(INPUT));
    long seed = hasOption(RANDOM_SEED) ? Long.parseLong(getOption(RANDOM_SEED)) : System.currentTimeMillis();
    double testPercent = hasOption(TEST_PERCENT) ? Double.parseDouble(getOption(TEST_PERCENT)) : 0.15;
    double validationPercent = hasOption(VALIDATION_PERCENT) ? Double.parseDouble(getOption(VALIDATION_PERCENT)) : 0.10;
    PrintWriter  trainOut = new PrintWriter(new File(getOption(TRAIN_OUT)));
    PrintWriter testOut = new PrintWriter(new File(getOption(TEST_OUT)));
    PrintWriter validationOut = new PrintWriter(new File(getOption(VALIDATION_OUT)));

    Random random = RandomUtils.getRandom(seed);
    for (String line : Files.readLines(input, Charsets.UTF_8)) {
      double d = random.nextDouble();
      if (d <= validationPercent) {
        validationOut.println(line);
      } else if (validationPercent < d && d <= validationPercent+testPercent) {
        testOut.println(line);
      } else {
        trainOut.println(line);
      }
    }
    return 0;
  }
}
