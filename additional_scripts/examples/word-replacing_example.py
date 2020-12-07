reviewList = [
    "Sometimes we bring the story to you, sometimes you have to go to the story.<br /><br />Alas no one listened, but that does not mean it should not have been said.",
    "Bromwell High is nothing short of brilliant. Expertly scripted and perfectly delivered, this searing parody of a students and teachers at a South London Public School leaves you literally rolling with laughter. It's vulgar, provocative, witty and sharp. The characters are a superbly caricatured cross section of British society (or to be more accurate, of any society). Following the escapades of Keisha, Latrina and Natella, our three 'protagonists' for want of a better term, the show doesn't shy away from parodying every imaginable subject. Political correctness flies out the window in every episode. If you enjoy shows that aren't afraid to poke fun of every taboo subject imaginable, then Bromwell High will not disappoint!"
]

newList = []
for line in reviewList:
    newList.append(line.replace("<br />", "\n"))
    
for newLine in newList:
    print(newLine)