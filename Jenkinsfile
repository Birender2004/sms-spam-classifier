pipeline{
    agent any
    
    stages{
        stage("Test"){
            steps{
                sh 'docker ps'
                sh 'pwd'
                sh 'ls'
            }
        }
        
        stage("Build"){
            steps{
                sh '''
                docker build -t demo-app:latest .
                
                '''
            }
        }
        stage("tagging"){
            steps{
                sh 'docker tag demo-app:latest birender2026/exp-spam-app:v1'
            }
        }
        
        stage("login"){
            steps{
                withCredentials([usernanmePassword(credentialsId:"new_creds", usernamevariable: "user", passwordVariable: "pass")]){
                    sh 'echo $pass | docker login -u $user --password-stdin'
                    echo 'login successfull'
                }
            }
        }
        stage("push"){
            steps{
                sh 'docker push birender2026/exp-spam-app:v1'
            }
        }
    }
    
    post{
        success{
            echo "Successssssssssssss"
        }
        failure{
            echo "Failureeeeeeeeeee"
        }
    }
}
