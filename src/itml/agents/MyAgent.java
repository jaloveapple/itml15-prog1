package itml.agents;

import itml.cards.Card;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;


import java.util.ArrayList;


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
        double[] values = new double[8];
        StateAgent a = stateBattle.getAgentState(0);
        StateAgent o = stateBattle.getAgentState(1);
        values[0] = a.getCol();
        values[1] = a.getRow();
        values[2] = a.getHealthPoints();
        values[3] = a.getStaminaPoints();
        values[4] = o.getCol();
        values[5] = o.getRow();
        values[6] = o.getHealthPoints();
        values[7] = o.getStaminaPoints();
        try {
            ArrayList<Card> allCards = m_deck.getCards();
            ArrayList<Card> cards = m_deck.getCards(a.getStaminaPoints());
            Instance i = new Instance(1.0, values.clone());
            i.setDataset(dataset);
            int out = (int)classifier_.classifyInstance(i);
            Card selected = allCards.get(out);
            if(cards.contains(selected)) {
                return selected;
            }
        } catch (Exception e) {
            System.out.println("Error classifying new instance: " + e.toString());
        }
        return new CardRest();  //To change body of implemented methods use File | Settings | File Templates.
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

    private Card randomMove( StateBattle stateBattle ){
        StateAgent stateAgent = stateBattle.getAgentState( m_noThisAgent );
        ArrayList<Card> cards = m_deck.getCards( stateAgent.getStaminaPoints() );
        Random m_random = new Random();
        return cards.get( m_random.nextInt( cards.size() ) );
    }

    private Card aggressiveMove( StateBattle stateBattle ){
        StateAgent asThis = stateBattle.getAgentState( m_noThisAgent );
        StateAgent asOpp  = stateBattle.getAgentState( m_noOpponentAgent );

        ArrayList<Card> cards = m_deck.getCards( asThis.getStaminaPoints() );

        // First check to see if we are in attack range, if so attack.
        for ( Card card : cards ) {
            if ( (card.getType() == Card.CardActionType.ctAttack) &&
                    card.inAttackRange( asThis.getCol(), asThis.getRow(),
                            asOpp.getCol(), asOpp.getRow() ) ) {
                return card;  // attack!
            }
        }

        // If we cannot attack, then try to move closer to the agent.
        Card [] move = new Card[2];
        move[m_noOpponentAgent] = new CardRest();

        Card bestCard = new CardRest();
        int  bestDistance = calcDistanceBetweenAgents( stateBattle );

        // ... otherwise move closer to the opponent.
        for ( Card card : cards ) {
            StateBattle bs = (StateBattle) stateBattle.clone();   // close the state, as play( ) modifies it.
            move[m_noThisAgent] = card;
            bs.play( move );
            int  distance = calcDistanceBetweenAgents( bs );
            if ( distance < bestDistance ) {
                bestCard = card;
                bestDistance = distance;
            }
        }

        return bestCard;
    }

    private int calcDistanceBetweenAgents( StateBattle stateBattle ) {

        StateAgent asFirst = stateBattle.getAgentState( 0 );
        StateAgent asSecond = stateBattle.getAgentState( 1 );

        return Math.abs( asFirst.getCol() - asSecond.getCol() ) + Math.abs( asFirst.getRow() - asSecond.getRow() );
    }
}
