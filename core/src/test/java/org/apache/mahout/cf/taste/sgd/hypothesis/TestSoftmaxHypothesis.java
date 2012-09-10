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

package org.apache.mahout.cf.taste.sgd.hypothesis;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Before;
import org.junit.Test;

public class TestSoftmaxHypothesis extends TasteTestCase{
  Hypothesis hypothesis;

  @Before
  public void setUp(){
    hypothesis = new SoftmaxHypothesis();
  }

  @Test
  public void predict(){
    double[] linearCombination = new double[] {3, 10, 2};
    double[] linearCombination1 = new double[] {2, 3, 10};
    double[] linearCombination2 = new double[] {1, 0, 2};

    double[] probs = hypothesis.predict(linearCombination);
    assertTrue(probs[1] > probs[0]);
    assertTrue(probs[0] > probs[2]);

    assertEquals(1.0, probs[0] + probs[1] + probs[2], 0.001);

    probs = hypothesis.predict(linearCombination1);
    assertTrue(probs[2]>probs[1]);
    assertTrue(probs[1]>probs[0]);
    assertEquals(1.0, probs[0]+probs[1]+probs[2], 0.001);

    probs = hypothesis.predict(linearCombination2);
    assertTrue(probs[2]>probs[0]);
    assertTrue(probs[0]>probs[1]);
    assertEquals(1.0, probs[0] + probs[1] + probs[2], 0.001);

    double[] invalidLinearCombination = new double[] {1, 2};
    double[] invalidLinearCombination1 = null;

    try{
      probs = hypothesis.predict(invalidLinearCombination);
      assertTrue("Should have thrown exception", false);
    }
    catch(IllegalArgumentException e){
      assertTrue(true);
    }
    catch(Exception e){
      assertTrue("Should have thrown IllegalArgumentException", false);
    }


    try{
      probs = hypothesis.predict(invalidLinearCombination1);
      assertTrue("Should have thrown exception", false);
    }
    catch(NullPointerException e){
      assertTrue(true);
    }
    catch(Exception e){
      assertTrue("Should have thrown NullPointerException", false);
    }
  }
}
