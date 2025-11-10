The "board game" consists of a few simple elements, a 5x5 grid world, a hero(main character/object) and a goal. 
Each grid hex contains one of the following elements: the hero, the crystal (goal), empty tiles, walls, monsters, potions, a key and a door. 
To further add to the resource management and allocation, an energy system is also implemented where we are given a starting amount, decrementing on each move, replenishable by potions and further reduced by a monster encounter. 
Reaching the crystal before the energy depletes is the main goal of the puzzle. 

The world is a 5×5 grid, and each cell has a meaning:

S is where the hero begins

  G -> the crystal

  . -> an empty tile that just costs 1 energy to cross

  W -> a wall you can’t pass

  M -> a monster that costs extra energy to deal with

  P -> a potion that restores some energy

  K -> the key

  D -> the locked door (needs the key and costs a bit more energy to move through)
