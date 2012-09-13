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

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.VectorWritable;

/**
 * With this class, one can prepare <Text, VectorWritable> sequence file from <Text, Text> side info file
 */
public class SideInfoSequenceToVector extends AbstractJob{

  public static String TO_VECTOR_CLASS = "toVectorClass";
  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(TO_VECTOR_CLASS, "tv", "Full class name of text to vector creator, which extends org.apache.mahout.cf.taste.sgd.common.ToVector", true);
    addOption(DefaultOptionCreator.helpOption());
    parseArguments(strings);

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    Configuration conf = getConf();
    Class<? extends ToVector> toVectorClass= Class.forName(getOption(TO_VECTOR_CLASS)).asSubclass(ToVector.class);

    SequenceFileIterator<Text, Text> iterator = new SequenceFileIterator<Text, Text>(inputPath, true, conf);
    SequenceFile.Writer writer = new SequenceFile.Writer(outputPath.getFileSystem(conf), conf, outputPath, Text.class, VectorWritable.class);
    while (iterator.hasNext()) {
      Pair<Text, Text> in = iterator.next();
      Text key = in.getFirst();
      VectorWritable value = new VectorWritable(toVectorClass.newInstance().get(in.getSecond()));
      writer.append(key, value);
    }

    Closeables.closeQuietly(writer);
    return 0;
  }
}
