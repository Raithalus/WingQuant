
for i = 1:412
CoordCheck(i,:) = length(intersect(bw16{1,i}, CoordArray.C3(:,:)));
end

figure,
plot(CoordCheck)

[ccx ccy] = max(CoordCheck)