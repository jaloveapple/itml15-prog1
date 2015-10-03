package itml.agents;

import itml.cards.Card;
import itml.cards.CardDefend;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import javax.swing.plaf.nimbus.State;
import java.util.Random;


import java.util.ArrayList;


/**
 *  This agent will demolish it's opponents. Not even the chosen one (Neo), can stand up to it.
 *
 *  @authors        Arnar Freyr Bjarnason
 *                  Kjartan Valur Kjartansson
 *                  Jón Gísli Björgvinsson
 */
public class MyAgent extends Agent{
    private int m_noThisAgent;     // Index of our agent (0 or 1).
    private int m_noOpponentAgent; // Index of opponent's agent.
    private Classifier classifier_;
    private Instances dataset;
    boolean learned;

    public MyAgent ( CardDeck deck, int msConstruct, int msPerMove, int msLearn){
        super(deck, msConstruct, msPerMove, msLearn);
        classifier_ = new J48();
        learned = false;
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
                return getMove(stateBattle, selected);
            }
        } catch (Exception e) {
            System.out.println("Error classifying new instance: " + e.toString());
        }
        return new CardRest();  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public Classifier learn( Instances instances ){
        learned = true;
        dataset = instances;
        try {
            classifier_.buildClassifier(instances);
        } catch (Exception e){
            System.out.println("Error training classifier: " + e.toString());
        }
        return null;
    }

    private Card chickenMove (StateBattle stateBattle ){
        Card [] move = new Card[2];

        move[m_noOpponentAgent] = new CardRest();   // We assume the opponent just stays where he/she is,
                                                    // and then take the move that brings us as far away as possible.

        Card bestCard = new CardRest();
        int  minDistance = calcDistanceBetweenAgents( stateBattle );

        ArrayList<Card> cards = m_deck.getCards( stateBattle.getAgentState( m_noThisAgent ).getStaminaPoints() );
        for ( Card card : cards ) {
            StateBattle bs = (StateBattle) stateBattle.clone();   // close the state, as play( ) modifies it.
            move[m_noThisAgent] = card;
            bs.play( move );
            int  distance = calcDistanceBetweenAgents( bs );
            if ( distance > minDistance ) {
                bestCard = card;
                minDistance = distance;
            }
        }

        return bestCard;
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

    /**
     * Calculate current score based on how long they can stay in close combat with each other
     * (currently assumes only attack and rest, not defend)
     *
     * @param stateBattle       Current state of game
     *
     * @return a number indicating which agent has a better position
     */

    private double score ( StateBattle stateBattle ){
        StateAgent agent = stateBattle.getAgentState(m_noThisAgent);
        StateAgent oAgent = stateBattle.getAgentState(m_noOpponentAgent);



        double turns, oturns;
        turns = agent.getHealthPoints() + Math.ceil((agent.getHealthPoints() - oAgent.getStaminaPoints())/3);
        oturns = oAgent.getHealthPoints() + Math.ceil((oAgent.getHealthPoints() - agent.getStaminaPoints())/3);
        return turns - oturns;
    }

    /**
     * Get our next move, based on that the opponents move is oCard
     *
     * @param stateBattle       Current state of game
     * @param oCard             Predicted opponent move
     *
     * @return
     */

    private Card getMove( StateBattle stateBattle, Card oCard){
        StateAgent agent = stateBattle.getAgentState(m_noThisAgent);
        StateAgent oagent = stateBattle.getAgentState(m_noOpponentAgent);

        // Agent is exhausted
        if(agent.getStaminaPoints() == 0) return new CardRest();

        // Other agent is defending
        if(oCard.getName().equals("cDefend")) return new CardRest();




        return new CardRest();
    }

    private int calcDistanceBetweenAgents( StateBattle stateBattle ) {

        StateAgent asFirst = stateBattle.getAgentState( 0 );
        StateAgent asSecond = stateBattle.getAgentState( 1 );

        return Math.abs( asFirst.getCol() - asSecond.getCol() ) + Math.abs( asFirst.getRow() - asSecond.getRow() );
    }
}
