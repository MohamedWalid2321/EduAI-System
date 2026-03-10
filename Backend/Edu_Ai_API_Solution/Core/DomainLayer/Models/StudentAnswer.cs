using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Models
{
    public class StudentAnswer:BaseEntity<int>  
    {
        public int QuizAttemptId { get; set; } // foreign key to the QuizAttempt entity
        public QuizAttempt QuizAttempt { get; set; } // navigation property to the QuizAttempt entity
        public int QuizQuestionId { get; set; } // foreign key to the QuizQuestion entity
        public QuizQuestion QuizQuestion { get; set; } // navigation property to the QuizQuestion entity
        public int QuestionChoiceId { get; set; } // foreign key to the QuestionChoices entity
        public QuestionChoices QuestionChoice { get; set; } // navigation property to the QuestionChoices entity
        public   bool IsCorrect { get; set; } // indicates whether the student's answer is correct or not
    }
}
