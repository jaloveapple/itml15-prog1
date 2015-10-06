package itml.cards;

import itml.simulator.Coordinate;

/**
 * Created by Arnar on 6.10.2015.
 */
public class CardAttackzDestroy extends Card {

    private static Coordinate[] destroyAttack = { coO, coE, coW };

    public CardAttackzDestroy() {
        super( "cAttackDestroy", Card.CardActionType.ctAttack, 0, 0, -3, 2, 0, destroyAttack );
    }
}
