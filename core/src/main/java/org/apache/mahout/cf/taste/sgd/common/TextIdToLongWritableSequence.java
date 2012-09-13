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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.VectorWritable;

/**
 * Because seq2sparse outs <Text, VectorWritable> pairs, this simple job converts Text keys to LongWritables.
 */
public class TextIdToLongWritableSequence extends AbstractJob {
  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.helpOption());
    parseArguments(strings);

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    Configuration conf = getConf();
    FileSystem fs = outputPath.getFileSystem(conf);
    SequenceFile.Writer writer = null;
    try {
      SequenceFileIterator<Text, VectorWritable> it = new SequenceFileIterator<Text, VectorWritable>(inputPath, true, conf);
      writer = new SequenceFile.Writer(fs, conf, outputPath, LongWritable.class, VectorWritable.class);
      while (it.hasNext()) {
        Pair<Text, VectorWritable> pair = it.next();
        LongWritable id = new LongWritable(Long.parseLong(pair.getFirst().toString()));
        VectorWritable vector = pair.getSecond();
        writer.append(id, vector);
      }
    } catch (Exception e) {
      return -1;
    } finally {
      Closeables.closeQuietly(writer);
    }
    return 0;
  }
}
