
<<<<<<< HEAD
for i = 1:412
CoordCheck(i,:) = length(intersect(bw16{1,i}, CoordArray.C3(:,:)));
=======
for i = 1:459
CoordCheck(i,:) = length(intersect(bw16{1,i}, CoordArray.Q4(:,:)));
>>>>>>> 55f423868254b78f82d706d5e6b1de85b5f2d171
end

figure,
plot(CoordCheck)

[ccx ccy] = max(CoordCheck)