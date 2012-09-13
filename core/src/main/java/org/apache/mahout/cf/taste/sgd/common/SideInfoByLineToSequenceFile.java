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
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

import java.io.File;

/**
 * A really simple class to convert a side info file of <id, side info> lines to <id, side info> sequence files with Text key and Text value
 * Later may be for creating sparse text vectors (see seq2sparse), or by {@link SideInfoSequenceToVector} for custom vectorization
 */
public class SideInfoByLineToSequenceFile extends AbstractJob{
  public static String SEPARATOR = "separator";

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(SEPARATOR, "sep", "String that separates id and side info text");
    addOption(DefaultOptionCreator.helpOption());
    parseArguments(strings);

    File inputFile = getInputFile();
    Path outputPath = getOutputPath();
    Configuration conf = new Configuration();

    SequenceFile.Writer seqWriter = new SequenceFile.Writer(outputPath.getFileSystem(conf), conf, outputPath, Text.class, Text.class);
    String separator = getOption(SEPARATOR);
    for(String line: Files.readLines(inputFile, Charsets.UTF_8)){
      String[] fields = line.split(separator);
      seqWriter.append(new Text(fields[0]), new Text(fields[1]));
    }
    Closeables.closeQuietly(seqWriter);
    return 0;
  }
}
