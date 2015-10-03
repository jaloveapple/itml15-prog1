package itml.agents;

import itml.cards.Card;
import itml.simulator.CardDeck;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;


/**
 *  This class ...
 *
 *  @authors        Arnar Freyr Bjarnason
 *                  *insert name*
 *                  *insert name*
 */
public class MyAgent extends Agent{
    private int m_noThisAgent;     // Index of our agent (0 or 1).
    private int m_noOpponentAgent; // Index of opponent's agent.
    private Classifier classifier_;
    private Instances dataset;

    public MyAgent ( CardDeck deck, int msConstruct, int msPerMove, int msLearn){
        super(deck, msConstruct, msPerMove, msLearn);
        classifier_ = new J48();
    }

    @Override
    public void startGame( int noThisAgent, StateBattle stateBattle ){
        m_noThisAgent = noThisAgent;
        m_noOpponentAgent = (noThisAgent == 0) ? 1 : 0;

    }

    @Override
    public void endGame( StateBattle stateBattle, double [] results ){

    }

    @Override
    public Card act( StateBattle stateBattle ){
        return null;
    }

    @Override
    public Classifier learn( Instances instances ){
        dataset = instances;
        try {
            classifier_.buildClassifier(instances);
        } catch (Exception e){
            System.out.println("Error training classifier: " + e.toString());
        }
        return null;
    }
}
