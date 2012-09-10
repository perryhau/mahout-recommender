/*
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

package org.apache.mahout.cf.taste.sgd.hypothesis;

import com.google.common.base.Preconditions;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;

import java.util.List;

/**
 * The hypothesis that does the actual rating prediction. Implementations should provide the distribution-specific predictions
 */
public abstract class Hypothesis {

  /**
   * given parameter and feature vectors, makes the prediction
   * @param x feature vector(s)
   * @param theta parameter vectors
   * @return the prediction of the hypothesis
   */
  public double[] predict (Vector[] x, Vector[] theta){
    Preconditions.checkArgument(x.length == theta.length);
    double [] linearCombination = new double[theta.length];

    for(int i = 0; i<linearCombination.length; i++){
      linearCombination[i] = x[i].dot(theta[i]);
    }
    return predict(linearCombination);
  }

  /**
   * given cuts, factors, side info parameters and features, computes the linearCombination
   * @param cuts this is not used
   * @param factors alphas and betas
   * @param others side info vectors and corresponding parameters
   * @return the linear combination (array of sum of dot products)
   */
  public double[] linearCombination(double[] cuts, Pair<Vector[], Vector[]> factors, List<Pair<Vector[], Vector>> others){
    Vector[] alphas = factors.getFirst();
    double[] linearCombination = new double[alphas.length];

    for(int i = 0; i<linearCombination.length; i++){
      linearCombination[i] = factors.getFirst()[i].dot(factors.getSecond()[i]);
      for(Pair<Vector[], Vector> pair:others){
        linearCombination[i] += pair.getFirst()[i].dot(pair.getSecond());
      }
    }
    return linearCombination;
  }

  /**
   * given cuts and factors, computes the linearCombination
   * @param cuts this is not used
   * @param alphas user factors
   * @param betas item factors
   * @return the linearCombination (array of dot products of alphas and betas)
   */
  public double[] linearCombination(double[] cuts, Vector[] alphas, Vector[] betas){
    double[] linearCombination = new double[alphas.length];

    for(int c = 0; c<linearCombination.length; c++){
      linearCombination[c] = alphas[c].dot(betas[c]);
    }
    return linearCombination;
  }

  /**
   * given linearCombination, computes the prediction of the specific hypothesis
   * @param linearCombination the linear combination of params and features
   * @return the prediction
   */
  public abstract double[] predict (double[] linearCombination);

  /**
   * given linearCombination, computes the whole score distribution for all classes
   * @param linearCombination the linear combination of params and features
   * @return the prediction
   */
  public abstract double[] predictFull(double[] linearCombination);

}
