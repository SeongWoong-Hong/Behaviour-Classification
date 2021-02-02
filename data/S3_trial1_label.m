% 0:Stay, 1:Walking, 2:Swing, 3:Transition
S3_trial1 = 3*ones(30400,1);
S3_trial1(44:2480) = 0;
S3_trial1(2480:3736)=1;
S3_trial1(4262:4519)=2;
S3_trial1(4838:6618)=1;
S3_trial1(7003:7401)=2;
S3_trial1(7575:9243)=1;
S3_trial1(9637:9906)=2;
S3_trial1(10160:11320)=1;
S3_trial1(11420:12390)=0;
S3_trial1(12510:15040)=1;
S3_trial1(15100:16250)=0;
S3_trial1(16350:17530)=1;
S3_trial1(17920:18230)=2;
S3_trial1(18530:20190)=1;
S3_trial1(20540:20880)=2;
S3_trial1(21090:22560)=1;
S3_trial1(23060:23390)=2;
S3_trial1(23650:24740)=1;
S3_trial1(24880:26240)=0;
S3_trial1(26240:28790)=1;
S3_trial1(28790:30170)=0;
save S3_trial1_label.txt S3_trial1 -ascii -tabs -double